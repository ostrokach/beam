# -*- coding: utf-8 -*-
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import absolute_import
from __future__ import print_function

import functools
import itertools
import os
import pickle
import time
import unittest
import uuid

import dill
from parameterized import parameterized

from apache_beam import transforms
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.pipeline import Pipeline
from apache_beam.runners.direct.direct_runner import BundleBasedDirectRunner
from apache_beam.runners.interactive.caching import pcollectioncache_it_test
from apache_beam.runners.interactive.caching import streambasedcache
from apache_beam.transforms import trigger
from apache_beam.transforms import window

# Protect against environments where the PubSub library is not available.
try:
  from google.cloud import pubsub
except ImportError:
  pubsub = None

READ_TIMEOUT = 60


def read_through_pipeline(cache, limit=None, timeout=None):
  """Read elements from cache using a Beam pipeline."""
  temp_location = "projects/{}/topics/read-test-{}".format(
      os.getenv("PUBSUB_PROJECT_ID", "test-project"),
      uuid.uuid4().hex)
  temp_cache = streambasedcache.PubSubBasedCache(temp_location, single_use=True)
  p = Pipeline(runner=BundleBasedDirectRunner(),
               options=PipelineOptions(streaming=True))
  # yapf: disable
  _ = (
      p
      | "Read" >> cache.reader()
      | "Window" >> transforms.WindowInto(
          window.GlobalWindows(),
          # beam.window.FixedWindows(1),
          trigger=trigger.Repeatedly(trigger.AfterCount(1)),
          accumulation_mode=trigger.AccumulationMode.DISCARDING)
      # | "Echo" >> transforms.Map(lambda e: print(e) or e)
      | "Write" >> temp_cache.writer()
  )
  # yapf: enable
  pr = p.run()
  try:
    results = list(itertools.islice(temp_cache.read(timeout=timeout), limit))
  finally:
    pr.cancel()
    temp_cache.remove()
  return results


def write_through_pipeline(cache, data_in, timeout=None):
  """Write elements to cache using a Beam pipeline."""
  temp_location = "projects/{}/topics/write-test-{}".format(
      os.getenv("PUBSUB_PROJECT_ID", "test-project"),
      uuid.uuid4().hex)
  temp_cache = streambasedcache.PubSubBasedCache(temp_location, single_use=True)
  p = Pipeline(runner=BundleBasedDirectRunner(),
               options=PipelineOptions(streaming=True))
  # yapf: disable
  pcoll = (
      p
      | "Read" >> transforms.Create(data_in)
      | "Window" >> transforms.WindowInto(
          window.GlobalWindows(),
          # beam.window.FixedWindows(1),
          trigger=trigger.Repeatedly(trigger.AfterCount(1)),
          accumulation_mode=trigger.AccumulationMode.DISCARDING)
      # | "Echo" >> transforms.Map(lambda e: print(e) or e)
  )
  _ = pcoll |  "Write" >> cache.writer()
  _ = pcoll |  "Write temp" >> temp_cache.writer()
  # yapf: enable
  pr = p.run()
  try:
    _ = list(itertools.islice(temp_cache.read(timeout=timeout), len(data_in)))
    time.sleep(0.2)  # In case the other branch is lagging behind.
  finally:
    pr.cancel()
    temp_cache.remove()


def read_directly(cache, limit=None, timeout=None):
  """Read elements from cache using the cache API."""
  return list(itertools.islice(cache.read(timeout=timeout), limit))


def write_directly(cache, data_in, timeout=None):
  """Write elements to cache using the cache API."""
  cache.write(data_in)


def retry_flakes(fn, max_tries=3):

  @functools.wraps(fn)
  def retry_fn(*args, **kwargs):
    num_tries = 0
    while True:
      try:
        return fn(*args, **kwargs)
      except AssertionError:
        num_tries += 1
        if num_tries >= max_tries:
          raise

  return retry_fn


class SerializationTestBase(pcollectioncache_it_test.SerializationTestBase):

  @parameterized.expand([("pickle", pickle), ("dill", dill)])
  def test_serde_empty(self, _, serializer):
    retry_flakes(self.check_serde_empty)(write_directly, read_directly,
                                         serializer)

  @parameterized.expand([("pickle", pickle), ("dill", dill)])
  def test_serde_filled(self, _, serializer):
    retry_flakes(self.check_serde_filled)(write_directly, read_directly,
                                          serializer)


@unittest.skipIf(pubsub is None, 'GCP dependencies are not installed.')
@unittest.skipIf(
    "PUBSUB_PROJECT_ID" not in os.environ,
    "We need to set the PUBSUB_PROJECT_ID environment variable to test against "
    "Goolgle Cloud PubSub.")
class PubSubBasedCacheSerializationTest(SerializationTestBase,
                                        unittest.TestCase):

  cache_class = streambasedcache.PubSubBasedCache

  def setUp(self):
    self.location = "projects/{}/topics/{}-{}".format(
        os.getenv("PUBSUB_PROJECT_ID", "test-project"),
        self.cache_class.__name__,
        uuid.uuid4().hex)

  def tearDown(self):
    streambasedcache.remove_topic_and_descendants(self.location)

  def get_writer_kwargs(self, data=None):
    return {"single_use": True}


class RoundtripTestBase(pcollectioncache_it_test.RoundtripTestBase):

  @parameterized.expand([
      ("{}-{}-{}".format(
          data_name,
          write_fn.__name__,
          read_fn.__name__,
      ), write_fn, read_fn, data)
      for data_name, data in pcollectioncache_it_test.GENERIC_TEST_DATA
      for write_fn in [write_directly, write_through_pipeline]
      for read_fn in [read_directly, read_through_pipeline]
  ])
  def test_roundtrip(self, _, write_fn, read_fn, data):
    retry_flakes(self.check_roundtrip)(write_fn, read_fn, data)


@unittest.skipIf(pubsub is None, 'GCP dependencies are not installed.')
@unittest.skipIf(
    "PUBSUB_PROJECT_ID" not in os.environ,
    "We need to set the PUBSUB_PROJECT_ID environment variable to test against "
    "Goolgle Cloud PubSub.")
@unittest.skipIf("PUBSUB_EMULATOR_HOST" in os.environ,
                 "PubSub emulators do not support snapshotting.")
class PubSubBasedCacheRoundtripTest(RoundtripTestBase, unittest.TestCase):

  cache_class = streambasedcache.PubSubBasedCache

  def setUp(self):
    self.location = "projects/{}/topics/{}-{}".format(
        os.getenv("PUBSUB_PROJECT_ID", "test-project"),
        self.cache_class.__name__,
        uuid.uuid4().hex)

  def tearDown(self):
    streambasedcache.remove_topic_and_descendants(self.location)


@unittest.skipIf(pubsub is None, 'GCP dependencies are not installed.')
@unittest.skipIf(
    "PUBSUB_EMULATOR_HOST" not in os.environ,
    "Not using an emulator, so running PubSubBasedCacheRoundtripTest instead.")
class FastPubSubBasedCacheRoundtripTest(
    pcollectioncache_it_test.ExtraAssertions, unittest.TestCase):

  cache_class = streambasedcache.PubSubBasedCache

  def setUp(self):
    self.location = "projects/{}/topics/{}-{}".format(
        os.getenv("PUBSUB_PROJECT_ID", "test-project"),
        self.cache_class.__name__,
        uuid.uuid4().hex)

  def tearDown(self):
    streambasedcache.remove_topic_and_descendants(self.location)

  def get_writer_kwargs(self, data=None):
    return {"single_use": True}

  def check_roundtrip(self, write_fn, read_fn, data):
    cache = self.cache_class(self.location, **self.get_writer_kwargs(data))
    write_fn(cache, data)
    data_out = read_fn(cache, limit=len(data))
    self.assertArrayCountEqual(data_out, data)

  @parameterized.expand([
      ("{}-{}-{}".format(
          data_name,
          write_fn.__name__,
          read_fn.__name__,
      ), write_fn, read_fn, data)
      for data_name, data in pcollectioncache_it_test.GENERIC_TEST_DATA
      for write_fn in [write_directly, write_through_pipeline]
      for read_fn in [read_directly, read_through_pipeline]
  ])
  def test_roundtrip(self, _, write_fn, read_fn, data):
    retry_flakes(self.check_roundtrip)(write_fn, read_fn, data)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
