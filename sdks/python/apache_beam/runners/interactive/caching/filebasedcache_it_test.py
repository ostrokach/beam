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

import logging
import os
import pickle
import shutil
import sys
import tempfile
import unittest
import uuid

import dill
import numpy as np
from parameterized import parameterized

from apache_beam import transforms
from apache_beam.io.filesystems import FileSystems
from apache_beam.runners.interactive.caching import datatype_inference
from apache_beam.runners.interactive.caching import filebasedcache
from apache_beam.runners.interactive.caching import pcollectioncache_it_test
from apache_beam.testing.test_pipeline import TestPipeline


def read_through_pipeline(cache, limit=None, timeout=None):
  """Read elements from cache using a Beam pipeline."""
  temp_dir = tempfile.mkdtemp()
  temp_cache = filebasedcache.SafeTextBasedCache(
      os.path.join(temp_dir,
                   uuid.uuid4().hex))
  try:
    with TestPipeline() as p:
      _ = (p | "Read" >> cache.reader() | "Write" >> temp_cache.writer())
    return list(temp_cache.read())
  finally:
    shutil.rmtree(temp_dir)


def write_through_pipeline(cache, data_in, timeout=None):
  """Write elements to cache using a Beam pipeline."""
  with TestPipeline() as p:
    _ = (p | "Create" >> transforms.Create(data_in) | "Write" >> cache.writer())


def read_directly(cache, limit=None, timeout=None):
  """Read elements from cache using the cache API."""
  return list(cache.read())


def write_directly(cache, data_in, timeout=None):
  """Write elements to cache using the cache API."""
  cache.write(data_in)


class SerializationTestBase(pcollectioncache_it_test.SerializationTestBase):

  def setUp(self):
    self._temp_dir = tempfile.mkdtemp()
    self.location = FileSystems.join(self._temp_dir, self.cache_class.__name__)

  def tearDown(self):
    FileSystems.delete([self._temp_dir])

  @parameterized.expand([("pickle", pickle), ("dill", dill)])
  def test_serde_empty(self, _, serializer):
    self.check_serde_empty(write_directly, read_directly, serializer)

  @parameterized.expand([("pickle", pickle), ("dill", dill)])
  def test_serde_filled(self, _, serializer):
    self.check_serde_filled(write_directly, read_directly, serializer)


class TextBasedCacheSerializationTest(SerializationTestBase, unittest.TestCase):

  cache_class = filebasedcache.TextBasedCache


class SafeTextBasedCacheSerializationTest(SerializationTestBase,
                                          unittest.TestCase):

  default_coder = filebasedcache.SafeFastPrimitivesCoder()
  cache_class = filebasedcache.SafeTextBasedCache


class TFRecordBasedCacheSerializationTest(SerializationTestBase,
                                          unittest.TestCase):

  cache_class = filebasedcache.TFRecordBasedCache


class AvroBasedCacheSerializationTest(SerializationTestBase, unittest.TestCase):

  cache_class = filebasedcache.AvroBasedCache

  def get_writer_kwargs(self, data=None):
    if sys.version_info > (3,):
      self.skipTest("Only fastavro is supported on Python 3.")
    use_fastavro = False
    schema = datatype_inference.infer_avro_schema(data,
                                                  use_fastavro=use_fastavro)
    return dict(schema=schema, use_fastavro=use_fastavro)


class FastAvroBasedCacheSerializationTest(SerializationTestBase,
                                          unittest.TestCase):

  cache_class = filebasedcache.AvroBasedCache

  def get_writer_kwargs(self, data=None):
    use_fastavro = True
    schema = datatype_inference.infer_avro_schema(data,
                                                  use_fastavro=use_fastavro)
    return dict(schema=schema, use_fastavro=use_fastavro)


class ParquetBasedCacheSerializationTest(SerializationTestBase,
                                         unittest.TestCase):

  cache_class = filebasedcache.ParquetBasedCache

  def get_writer_kwargs(self, data=None):
    schema = datatype_inference.infer_pyarrow_schema(data)
    return dict(schema=schema)


class RoundtripTestBase(pcollectioncache_it_test.RoundtripTestBase):

  def setUp(self):
    self._temp_dir = tempfile.mkdtemp()
    self.location = FileSystems.join(self._temp_dir, self.cache_class.__name__)

  def tearDown(self):
    FileSystems.delete([self._temp_dir])


class DatabagRoundtripTestBase(RoundtripTestBase):

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
    return self.check_roundtrip(write_fn, read_fn, data)


class DataframeRoundtripTestBase(RoundtripTestBase):

  @parameterized.expand([
      ("{}-{}-{}".format(
          data_name,
          write_fn.__name__,
          read_fn.__name__,
      ), write_fn, read_fn, data)
      for data_name, data in pcollectioncache_it_test.DATAFRAME_TEST_DATA
      for write_fn in [write_directly, write_through_pipeline]
      for read_fn in [read_directly, read_through_pipeline]
  ])
  def test_roundtrip(self, _, write_fn, read_fn, data):
    return self.check_roundtrip(write_fn, read_fn, data)


class TextBasedCacheRoundtripTest(DatabagRoundtripTestBase, unittest.TestCase):

  cache_class = filebasedcache.TextBasedCache

  def check_roundtrip(self, write_fn, read_fn, data):
    if data == [{"a": 10}]:
      self.skipTest(
          "TextBasedCache crashes for this particular case. "
          "One of the reasons why it should not be used in production.")
    if any(isinstance(e, np.ndarray) for e in data):
      self.skipTest("Numpy arrays are not supported.")
    return super(TextBasedCacheRoundtripTest,
                 self).check_roundtrip(write_fn, read_fn, data)


class SafeTextBasedCacheRoundtripTest(DatabagRoundtripTestBase,
                                      unittest.TestCase):

  cache_class = filebasedcache.SafeTextBasedCache


class TFRecordBasedCacheRoundtripTest(DatabagRoundtripTestBase,
                                      unittest.TestCase):

  cache_class = filebasedcache.TFRecordBasedCache


class AvroBasedCacheRoundtripBase(DataframeRoundtripTestBase):

  cache_class = filebasedcache.AvroBasedCache

  def check_roundtrip(self, write_fn, read_fn, data):
    if not data:
      pass
    elif not all(
        (sorted(data[0].keys()) == sorted(d.keys()) for d in data[1:])):
      self.skipTest("Rows with missing columns are not supported.")
    elif any((isinstance(v, np.ndarray) for v in data[0].values())):
      self.skipTest("Array data are not supported.")
    return super(AvroBasedCacheRoundtripBase,
                 self).check_roundtrip(write_fn, read_fn, data)


class AvroBasedCacheRoundtripTest(AvroBasedCacheRoundtripBase,
                                  unittest.TestCase):

  def get_writer_kwargs(self, data=None):
    if sys.version_info > (3,):
      self.skipTest("Only fastavro is supported on Python 3.")
    use_fastavro = False
    schema = datatype_inference.infer_avro_schema(data,
                                                  use_fastavro=use_fastavro)
    return dict(schema=schema, use_fastavro=use_fastavro)


class FastAvroBasedCacheRoundtripTest(AvroBasedCacheRoundtripBase,
                                      unittest.TestCase):

  def get_writer_kwargs(self, data=None):
    use_fastavro = True
    schema = datatype_inference.infer_avro_schema(data,
                                                  use_fastavro=use_fastavro)
    return dict(schema=schema, use_fastavro=use_fastavro)


class ParquetBasedCacheRoundtripTest(DataframeRoundtripTestBase,
                                     unittest.TestCase):

  cache_class = filebasedcache.ParquetBasedCache

  def get_writer_kwargs(self, data=None):
    schema = datatype_inference.infer_pyarrow_schema(data)
    return dict(schema=schema)

  def check_roundtrip(self, write_fn, read_fn, data):
    if not data:
      self.skipTest("Empty PCollections are not supported.")
    elif not data[0]:
      self.skipTest("PCollections with no columns are not supported.")
    elif not all(
        (sorted(data[0].keys()) == sorted(d.keys()) for d in data[1:])):
      self.skipTest("Rows with missing columns are not supported.")
    return super(ParquetBasedCacheRoundtripTest,
                 self).check_roundtrip(write_fn, read_fn, data)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
