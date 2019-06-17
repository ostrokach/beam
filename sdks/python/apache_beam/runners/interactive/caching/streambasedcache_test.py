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

import time
import unittest

import mock
from parameterized import parameterized

import apache_beam as beam
from apache_beam.io.gcp.pubsub import _PubSubSink
from apache_beam.pipeline import Pipeline
from apache_beam.pvalue import PBegin
from apache_beam.pvalue import PCollection
from apache_beam.runners.interactive.caching import filebasedcache_it_test
from apache_beam.runners.interactive.caching import pcollectioncache_it_test
from apache_beam.runners.interactive.caching import streambasedcache

# Protect against environments where the PubSub library is not available.
try:
  from google.cloud import pubsub_v1  # pylint: disable=ungrouped-imports
  from google.api_core import exceptions as gexc
except ImportError:
  pubsub_v1 = None
  gexc = None


class FakePublisherClient(object):
  """A class to mock basic Google Cloud PubSub CRUD operations."""

  def __init__(self, topics, subscriptions, snapshots):
    self._topics = topics
    self._subscriptions = subscriptions
    self._snapshots = snapshots

  def __setattr__(self, item, value):
    protected_attributes = ["_topics", "_subscriptions", "_snapshots"]
    if item in protected_attributes and hasattr(self, item):
      raise ValueError("Attributes {} should not be overwritten.".format(
          protected_attributes))
    super(FakePublisherClient, self).__setattr__(item, value)

  def create_topic(self, name, *args, **kwargs):
    if name in self._topics:
      raise gexc.AlreadyExists("Topic '{}' already exists.".format(name))
    self._topics.append(name)

  def delete_topic(self, topic, *args, **kwargs):
    if topic not in self._topics:
      raise gexc.NotFound("Topic '{}' was not found.".format(topic))
    self._topics.remove(topic)

  def get_topic(self, topic, *args, **kwargs):
    if topic not in self._topics:
      raise gexc.NotFound("Topic '{}' was not found.".format(topic))
    topic_proto = pubsub_v1.types.Topic()
    topic_proto.name = topic
    return topic_proto

  def list_topics(self, project, *args, **kwargs):
    topic_protos = []
    for topic in self._topics:
      topic_project = "/".join(topic.split("/")[:2])
      if topic_project == project:
        topic_proto = pubsub_v1.types.Topic()
        topic_proto.name = topic
        topic_protos.append(topic_proto)
    return topic_protos

  def list_topic_subscriptions(self, topic, *args, **kwargs):
    topic_project = topic.split("/")[1]
    subscriptions = []
    for sub_name, _ in self._subscriptions:
      sub_project = sub_name.split("/")[1]
      if sub_project == topic_project:
        subscriptions.append(sub_name)
    return subscriptions

  def topic_path(self, project, topic_name):
    return "projects/{}/topics/{}".format(project, topic_name)

  def publish(self, *args, **kwargs):

    class Future:

      def __init__(self, result):
        self._result = result

      def result(self, *args, **kwargs):
        return self._result

    return Future(None)


class FakeSubscriberClient(object):

  def __init__(self, topics, subscriptions, snapshots):
    self._topics = topics
    self._subscriptions = subscriptions
    self._snapshots = snapshots

  def __setattr__(self, item, value):
    protected_attributes = ["_topics", "_subscriptions", "_snapshots"]
    if item in protected_attributes and hasattr(self, item):
      raise ValueError("Attributes {} should not be overwritten.".format(
          protected_attributes))
    super(FakeSubscriberClient, self).__setattr__(item, value)

  def create_subscription(self, name, topic, *args, **kwargs):
    if name in [s for s, t in self._subscriptions]:
      raise gexc.AlreadyExists("Subscription '{}' already exists.".format(name))
    self._subscriptions.append((name, topic))

  def delete_subscription(self, subscription, *args, **kwargs):
    if subscription not in [s for s, t in self._subscriptions]:
      raise gexc.NotFound(
          "Subscription '{}' does not exist.".format(subscription))
    topic = next(t for s, t in self._subscriptions if s == subscription)
    self._subscriptions.remove((subscription, topic))

  def get_subscription(self, subscription, *args, **kwargs):
    for sub_name, sub_topic in self._subscriptions:
      if sub_name == subscription:
        subscription_proto = pubsub_v1.types.Subscription()
        subscription_proto.name = sub_name
        subscription_proto.topic = sub_topic
        return subscription_proto
    raise gexc.NotFound(
        "Subscription '{}' does not exist.".format(subscription))

  def list_subscriptions(self, project, *args, **kwargs):
    sub_protos = []
    for sub_name, _ in self._subscriptions:
      sub_project = "/".join(sub_name.split("/")[:2])
      if sub_project == project:
        sub_proto = pubsub_v1.types.Subscription()
        sub_proto.name = snap_name
        sub_proto.topic = snap_topic
        sub_proto.append(snap_proto)
    return sub_protos

  def create_snapshot(self, name, subscription, *args, **kwargs):
    if name in [s for s, t in self._snapshots]:
      raise gexc.AlreadyExists("Snapshot '{}' already exists.".format(name))
    topic = self.get_subscription(subscription).topic
    self._snapshots.append((name, topic))

  def delete_snapshot(self, snapshot, *args, **kwargs):
    if snapshot not in [s for s, t in self._snapshots]:
      raise gexc.NotFound("Snapshot '{}' does not exist.".format(snapshot))
    topic = next(t for s, t in self._snapshots if s == snapshot)
    self._snapshots.remove((snapshot, topic))

  def get_snapshot(self, snapshot, *args, **kwargs):
    # Blocked by: https://github.com/googleapis/google-cloud-python/issues/8554
    raise NotImplementedError

  def list_snapshots(self, project, *args, **kwargs):
    snap_protos = []
    for snap_name, snap_topic in self._snapshots:
      snap_project = "/".join(snap_name.split("/")[:2])
      if snap_project == project:
        snap_proto = pubsub_v1.types.Snapshot()
        snap_proto.name = snap_name
        snap_proto.topic = snap_topic
        snap_protos.append(snap_proto)
    return snap_protos

  def seek(self, subscription, time=None, snapshot=None, *kwargs):
    pass


class FakePubSub(object):

  def __init__(self):
    self._topics = []
    self._subscriptions = []
    self._snapshots = []

  def PublisherClient(self):
    return FakePublisherClient(self._topics, self._subscriptions,
                               self._snapshots)

  def SubscriberClient(self):
    return FakeSubscriberClient(self._topics, self._subscriptions,
                                self._snapshots)


@mock.patch(
    "apache_beam.runners.interactive.caching.streambasedcache.pubsub_v1",
    new_callable=FakePubSub)
class PubSubBasedCacheTest(pcollectioncache_it_test.ExtraAssertions,
                           unittest.TestCase):

  cache_class = streambasedcache.PubSubBasedCache

  def setUp(self):
    project_id = "test-project-id"
    topic_name = "test-topic-name"
    self.location = "projects/{}/topics/{}".format(project_id, topic_name)

  def tearDown(self):
    pass

  def test_init(self, mock_pubsub):
    _, project_id, _, topic_name = self.location.split("/")
    subscription_path = "projects/{}/subscriptions/{}".format(
        project_id, topic_name)
    snapshot_path = "projects/{}/snapshots/{}".format(project_id, topic_name)

    cache = self.cache_class(self.location)

    self.assertEqual(cache.location, self.location)
    self.assertEqual(cache.subscription_path, subscription_path)
    self.assertEqual(cache.snapshot_path, snapshot_path)

    self.assertCountEqual(mock_pubsub._topics, [self.location])
    self.assertCountEqual(mock_pubsub._subscriptions,
                          [(subscription_path, self.location)])
    self.assertCountEqual(mock_pubsub._snapshots,
                          [(snapshot_path, self.location)])

  def test_init_mode_invalid(self, unused_mock_pubsub):
    with self.assertRaises(ValueError):
      _ = self.cache_class(self.location, mode=None)

    with self.assertRaises(ValueError):
      _ = self.cache_class(self.location, mode="")

    with self.assertRaises(ValueError):
      _ = self.cache_class(self.location, mode="be happy")

  def test_init_mode_error(self, mock_pubsub):
    _ = self.cache_class(self.location)
    with self.assertRaises(IOError):
      _ = self.cache_class(self.location, mode="error")

  def test_init_mode_overwrite(self, mock_pubsub):
    location_copy = "".join(self.location)
    self.assertEqual(self.location, location_copy)
    self.assertNotEqual(id(self.location), id(location_copy))

    cache = self.cache_class(self.location)  # pylint: disable=unused-variable
    topic = mock_pubsub._topics[0]
    subscription = mock_pubsub._subscriptions[0]
    snapshot = mock_pubsub._snapshots[0]

    cache2 = self.cache_class(location_copy, mode="overwrite")  # pylint: disable=unused-variable
    self.assertCountEqual([topic], mock_pubsub._topics)
    self.assertCountEqual([subscription], mock_pubsub._subscriptions)
    self.assertCountEqual([snapshot], mock_pubsub._snapshots)
    self.assertEqual(id(topic), id(mock_pubsub._topics[0]))
    self.assertNotEqual(id(subscription), id(mock_pubsub._subscriptions[0]))
    self.assertNotEqual(id(snapshot), id(mock_pubsub._snapshots[0]))

  def test_init_mode_append(self, mock_pubsub):
    location_copy = "".join(self.location)
    self.assertEqual(self.location, location_copy)
    self.assertNotEqual(id(self.location), id(location_copy))

    cache = self.cache_class(self.location)  # pylint: disable=unused-variable
    topic = mock_pubsub._topics[0]
    subscription = mock_pubsub._subscriptions[0]
    snapshot = mock_pubsub._snapshots[0]

    cache2 = self.cache_class(location_copy, mode="append")  # pylint: disable=unused-variable
    self.assertCountEqual([topic], mock_pubsub._topics)
    self.assertCountEqual([subscription], mock_pubsub._subscriptions)
    self.assertCountEqual([snapshot], mock_pubsub._snapshots)
    self.assertEqual(id(topic), id(mock_pubsub._topics[0]))
    self.assertEqual(id(subscription), id(mock_pubsub._subscriptions[0]))
    self.assertEqual(id(snapshot), id(mock_pubsub._snapshots[0]))

  def test_timestamp(self, unused_mock_pubsub):
    cache = self.cache_class(self.location)
    self.assertEqual(cache.timestamp, 0)
    cache.writer()
    timestamp1 = cache.timestamp
    self.assertGreater(timestamp1, 0)
    time.sleep(0.01)
    cache.writer()
    timestamp2 = cache.timestamp
    self.assertGreater(timestamp2, timestamp1)

  def test_writer_arguments(self, unused_mock_pubsub):
    kwargs = {"a": 10, "b": "hello"}
    dummy_ptransfrom = beam.Map(lambda e: e)
    dummy_pcoll = PCollection(pipeline=Pipeline(), element_type=str)
    with mock.patch("apache_beam.runners.interactive.caching.streambasedcache"
                    ".PubSubBasedCache._writer_class") as mock_writer:
      mock_writer.return_value = dummy_ptransfrom
      cache = self.cache_class(self.location, **kwargs)
      cache.writer().expand(dummy_pcoll)
    _, kwargs_out = list(mock_writer.call_args)
    self.assertEqual(kwargs_out, kwargs)

  def test_reader_arguments(self, unused_mock_pubsub):

    def get_reader_kwargs(kwargs, passthrough):
      dummy_ptransfrom = beam.Map(lambda e: e)
      with mock.patch("apache_beam.runners.interactive.caching.streambasedcache"
                      ".PubSubBasedCache._reader_class") as mock_reader:
        mock_reader.return_value = dummy_ptransfrom
        cache = self.cache_class(self.location, mode="overwrite", **kwargs)
        cache._reader_passthrough_arguments = (
            cache._reader_passthrough_arguments | passthrough)
        cache.reader(coder=None).expand(PBegin(Pipeline()))
      _, kwargs_out = list(mock_reader.call_args)
      kwargs_out.pop("subscription")
      return kwargs_out

    kwargs = {"a": 10, "b": "hello world", "c": b"xxx"}
    kwargs_out = get_reader_kwargs(kwargs, set())
    self.assertEqual(kwargs_out, {})

    kwargs_out = get_reader_kwargs(kwargs, {"a", "b"})
    self.assertEqual(kwargs_out, {"a": 10, "b": "hello world"})

  @mock.patch("apache_beam.runners.interactive.caching.streambasedcache"
              ".PubSubBasedCache._writer_class")
  def test_writer(self, mock_writer, unused_mock_pubsub):
    """Test that a new writer is constructed each time `writer()` is called."""
    dummy_ptransfrom = beam.Map(lambda e: e)
    mock_writer.return_value = dummy_ptransfrom
    cache = self.cache_class(self.location)
    self.assertEqual(mock_writer.call_count, 0)
    cache.writer().expand(PCollection(Pipeline(), element_type=str))
    self.assertEqual(mock_writer.call_count, 1)
    cache.writer().expand(PCollection(Pipeline(), element_type=str))
    self.assertEqual(mock_writer.call_count, 2)

  @mock.patch("apache_beam.runners.interactive.caching.streambasedcache"
              ".PubSubBasedCache._reader_class")
  def test_reader(self, mock_reader, unused_mock_pubsub):
    """Test that a new reader is constructed each time `reader()` is called."""
    dummy_ptransfrom = beam.Map(lambda e: e)
    mock_reader.return_value = dummy_ptransfrom
    cache = self.cache_class(self.location)
    self.assertEqual(mock_reader.call_count, 0)
    cache.reader(coder=None).expand(PBegin(Pipeline()))
    self.assertEqual(mock_reader.call_count, 1)
    cache.reader(coder=None).expand(PBegin(Pipeline()))
    self.assertEqual(mock_reader.call_count, 2)

  def test_write(self, mock_pubsub):
    pass

  def test_read(self, mock_pubsub):
    pass

  def test_truncate(self, mock_pubsub):
    pass

  def test_remove(self, mock_pubsub):
    cache = self.cache_class(self.location)
    cache.remove()
    self.assertCountEqual([], mock_pubsub._topics)
    self.assertCountEqual([], mock_pubsub._subscriptions)
    self.assertCountEqual([], mock_pubsub._snapshots)

  @unittest.skip("Not implemented.")
  def test_garbage_collection(self):
    pass


@mock.patch(
    "apache_beam.runners.interactive.caching.streambasedcache.pubsub_v1",
    new_callable=FakePubSub)
class PubSubBasedCacheCoderTest(pcollectioncache_it_test.CoderTestBase,
                                unittest.TestCase):
  """Make sure that the coder gets set correctly when we write data to cache."""

  cache_class = streambasedcache.PubSubBasedCache

  def setUp(self):
    self.location = "projects/test-project-id/topics/test-topic-name"

  @parameterized.expand([
      ("{}-{}".format(data_name, write_fn.__name__), write_fn, data)
      for data_name, data in pcollectioncache_it_test.GENERIC_TEST_DATA
      for write_fn in [
          filebasedcache_it_test.write_directly,
          filebasedcache_it_test.write_through_pipeline
      ]
  ])
  def test_coder(self, mock_pubsub, _, write_fn, data):
    dummy_ptransfrom = beam.Map(lambda e: e)
    writer_kwargs = self.get_writer_kwargs(data)
    dummy_ptransfrom._sink = _PubSubSink(
        topic=self.location,
        id_label=writer_kwargs.get("id_label"),
        with_attributes=writer_kwargs.get("with_attributes"),
        timestamp_attribute=writer_kwargs.get("timestamp_attribute"))
    with mock.patch(
        "apache_beam.runners.interactive.caching.streambasedcache"
        ".PubSubBasedCache._writer_class") as mock_writer, \
        mock.patch("google.cloud.pubsub", new=mock_pubsub):
      mock_writer.return_value = dummy_ptransfrom
      self.check_coder(write_fn, data)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
