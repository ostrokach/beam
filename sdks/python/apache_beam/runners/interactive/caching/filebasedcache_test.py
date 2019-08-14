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

import gc
import itertools
import logging
import tempfile
import time
import unittest
import uuid

import mock
import numpy as np
from parameterized import parameterized

from apache_beam import coders
from apache_beam.io.filesystems import FileSystems
from apache_beam.runners.interactive.caching import filebasedcache
from apache_beam.testing import datatype_inference

GENERIC_TEST_DATA = [
    [],
    [None],
    [None, None, None],
    ["ABC", "DeF"],
    [u"ABC", u"±♠Ωℑ"],
    [b"abc123", b"aAaAa"],
    [1.5],
    [100, -123.456, 78.9],
    ["ABC", 1.2, 100, 0, -10, None, b"abc123"],
    [()],
    [("a", "b", "c")],
    [("a", 1, 1.2), ("b", 2, 5.5)],
    [("a", 1, 1.2), (2.5, "c", None)],
    [{}],
    [{"col1": "a", "col2": 1, "col3": 1.5}],
    [{"col1": "a", "col2": 1, "col3": 1.5},
     {"col1": "b", "col2": 2, "col3": 4.5}],
    [{"col1": "a", "col2": 1, "col3": u"±♠Ω"}, {4: 1, 5: 3.4, (6, 7): "a"}],
    [("a", "b", "c"), ["d", 1], {"col1": 1, 202: 1.234}, None, "abc", b"def",
     100, (1, 2, 3, "b")],
    [np.zeros((3, 6)), np.ones((10, 22))],
]


def read_directly(cache, limit=None, timeout=None):
  return list(itertools.islice(cache.read(), limit))


def write_directly(cache, data_in, timeout=None):
  """Write elements to cache using the cache API."""
  cache.write(data_in)


def write_through_pipeline(cache, data_in, timeout=None):
  """Write elements to cache using a Beam pipeline."""
  import apache_beam as beam

  with beam.Pipeline() as p:
    _ = (p | "Create" >> beam.Create(data_in) | "Write" >> cache.writer())


class FileBasedCacheTest(unittest.TestCase):

  def cache_class(self, location, *args, **kwargs):

    class MockedFileBasedCache(filebasedcache.FileBasedCache):

      _reader_class = mock.MagicMock()
      _writer_class = mock.MagicMock()
      _reader_passthrough_arguments = {}
      requires_coder = False

    return MockedFileBasedCache(location, *args, **kwargs)

  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()
    self.location = FileSystems.join(self.temp_dir, self.cache_class.__name__)

  def tearDown(self):
    FileSystems.delete([self.temp_dir])

  def create_dummy_file(self, location):
    """Create a dummy file with `location` as the filepath prefix."""
    filename = location + "-" + uuid.uuid4().hex
    while FileSystems.exists(filename):
      filename = location + "-" + uuid.uuid4().hex
    with open(filename, "wb") as fout:
      fout.write(b"dummy data")
    return filename

  def test_init(self):
    # cache needs to be kept so that resources are not gc-ed immediately.
    cache = self.cache_class(self.location)  # pylint: disable=unused-variable

    with self.assertRaises(ValueError):
      _ = self.cache_class(self.location, mode=None)

    with self.assertRaises(ValueError):
      _ = self.cache_class(self.location, mode="be happy")

  def test_overwrite_cache_0(self):
    # cache needs to be kept so that resources are not gc-ed immediately.
    cache = self.cache_class(self.location)  # pylint: disable=unused-variable
    # Refuse to create a cache with the same data storage location
    with self.assertRaises(IOError):
      _ = self.cache_class(self.location)

  def test_overwrite_cache_1(self):
    # cache needs to be kept so that resources are not gc-ed immediately.
    cache = self.cache_class(self.location)  # pylint: disable=unused-variable
    # Refuse to create a cache with the same data storage location
    _ = self.create_dummy_file(self.location)
    with self.assertRaises(IOError):
      _ = self.cache_class(self.location)

  def test_overwrite_cache_2(self):
    # cache needs to be kept so that resources are not gc-ed immediately.
    cache = self.cache_class(self.location)  # pylint: disable=unused-variable
    # OK to overwrite cache when in "overwrite" mode
    _ = self.cache_class(self.location, mode="overwrite")

  def test_overwrite_cache_3(self):
    cache = self.cache_class(self.location)
    # OK to overwrite cleared cache
    cache.remove()
    _ = self.cache_class(self.location)

  def test_persist_false_0(self):
    cache = self.cache_class(self.location, persist=False)
    _ = self.create_dummy_file(self.location)
    files = self._glob_files(self.location + "**")
    self.assertGreater(len(files), 0)
    cache.remove()
    files = self._glob_files(self.location + "**")
    self.assertEqual(len(files), 0)

  def test_persist_false_1(self):
    cache = self.cache_class(self.location, persist=False)
    _ = self.create_dummy_file(self.location)
    files = self._glob_files(self.location + "**")
    self.assertGreater(len(files), 0)
    del cache
    gc.collect()
    files = self._glob_files(self.location + "**")
    self.assertEqual(len(files), 0)

  def test_persist_true_0(self):
    cache = self.cache_class(self.location, persist=True)
    _ = self.create_dummy_file(self.location)
    files = self._glob_files(self.location + "**")
    self.assertGreater(len(files), 0)
    cache.remove()
    files = self._glob_files(self.location + "**")
    self.assertGreater(len(files), 0)

  def test_persist_true_1(self):
    cache = self.cache_class(self.location, persist=True)
    _ = self.create_dummy_file(self.location)
    files = self._glob_files(self.location + "**")
    self.assertGreater(len(files), 0)
    del cache
    gc.collect()
    files = self._glob_files(self.location + "**")
    self.assertGreater(len(files), 0)

  def _glob_files(self, pattern):
    files = [
        metadata.path
        for match in FileSystems.match([self.location + "**"])
        for metadata in match.metadata_list
    ]
    return files

  def test_timestamp(self):
    # Seems to always pass when delay=0.01 but set higher to prevent flakiness
    cache = self.cache_class(self.location)
    timestamp0 = cache.timestamp
    time.sleep(0.1)
    _ = self.create_dummy_file(self.location)
    timestamp1 = cache.timestamp
    self.assertGreater(timestamp1, timestamp0)

  def test_writer_arguments(self):
    kwargs = {"a": 10, "b": "hello"}
    cache = self.cache_class(self.location, **kwargs)
    cache.writer()
    _, kwargs_out = list(cache._writer_class.call_args)
    self.assertEqual(kwargs_out, kwargs)

  def test_reader_arguments(self):

    def check_reader_passthrough_kwargs(kwargs, passthrough):
      cache = self.cache_class(self.location, mode="overwrite", **kwargs)
      cache._reader_passthrough_arguments = passthrough
      cache.reader()
      _, kwargs_out = list(cache._reader_class.call_args)
      self.assertEqual(kwargs_out, {k: kwargs[k] for k in passthrough})

    check_reader_passthrough_kwargs({"a": 10, "b": "hello world"}, {})
    check_reader_passthrough_kwargs({"a": 10, "b": "hello world"}, {"b"})

  def test_writer(self):
    cache = self.cache_class(self.location)
    self.assertEqual(cache._writer_class.call_count, 0)
    cache.writer()
    self.assertEqual(cache._writer_class.call_count, 1)
    cache.writer()
    self.assertEqual(cache._writer_class.call_count, 2)

  def test_reader(self):
    cache = self.cache_class(self.location)
    self.assertEqual(cache._reader_class.call_count, 0)
    cache.reader()
    self.assertEqual(cache._reader_class.call_count, 1)
    cache.reader()
    self.assertEqual(cache._reader_class.call_count, 2)

  def test_write(self):
    cache = self.cache_class(self.location)
    self.assertEqual(cache._writer_class()._sink.open.call_count, 0)
    self.assertEqual(cache._writer_class()._sink.write_record.call_count, 0)
    self.assertEqual(cache._writer_class()._sink.close.call_count, 0)

    cache.write((i for i in range(11)))
    self.assertEqual(cache._writer_class()._sink.open.call_count, 1)
    self.assertEqual(cache._writer_class()._sink.write_record.call_count, 11)
    self.assertEqual(cache._writer_class()._sink.close.call_count, 1)

    cache.write((i for i in range(5)))
    self.assertEqual(cache._writer_class()._sink.open.call_count, 2)
    self.assertEqual(cache._writer_class()._sink.write_record.call_count, 16)
    self.assertEqual(cache._writer_class()._sink.close.call_count, 2)

    class DummyError(Exception):
      pass

    cache._writer_class()._sink.write_record.side_effect = DummyError
    with self.assertRaises(DummyError):
      cache.write((i for i in range(5)))
    self.assertEqual(cache._writer_class()._sink.open.call_count, 3)
    self.assertEqual(cache._writer_class()._sink.write_record.call_count, 17)
    self.assertEqual(cache._writer_class()._sink.close.call_count, 3)

  def test_read(self):
    cache = self.cache_class(self.location)
    _ = self.create_dummy_file(self.location)
    self.assertEqual(cache._reader_class()._source.read.call_count, 0)
    cache.read()
    # ._read does not get called unless we get items from iterator
    self.assertEqual(cache._reader_class()._source.read.call_count, 0)
    list(cache.read())
    self.assertEqual(cache._reader_class()._source.read.call_count, 1)
    list(cache.read())
    self.assertEqual(cache._reader_class()._source.read.call_count, 2)

  def test_truncate(self):
    cache = self.cache_class(self.location)
    self.assertEqual(
        len(list(filebasedcache.glob_files(cache.file_pattern))), 1)
    _ = self.create_dummy_file(self.location)
    self.assertEqual(
        len(list(filebasedcache.glob_files(cache.file_pattern))), 2)
    cache.truncate()
    self.assertEqual(
        len(list(filebasedcache.glob_files(cache.file_pattern))), 1)

  def test_clear_data(self):
    cache = self.cache_class(self.location)
    self.assertEqual(
        len(list(filebasedcache.glob_files(cache.file_pattern))), 1)
    _ = self.create_dummy_file(self.location)
    self.assertEqual(
        len(list(filebasedcache.glob_files(cache.file_pattern))), 2)
    cache.remove()
    self.assertEqual(
        len(list(filebasedcache.glob_files(cache.file_pattern))), 0)

  def test_clear_metadata(self):
    # Keep provided coder.
    cache = self.cache_class(self.location, coder="mock")
    self.assertEqual(cache.coder, "mock")
    cache.truncate()
    self.assertEqual(cache.coder, "mock")
    cache.remove()
    # Clean up inferred coder.
    cache = self.cache_class(self.location)
    self.assertEqual(cache.coder, None)
    cache.coder = "mock"
    cache.truncate()
    self.assertEqual(cache.coder, None)


class CoderTestBase(object):
  """Make sure that the coder gets set correctly when we write data to cache.

  Only applicable to implementations which infer the coder when writing data.
  """

  # Attributes to be set by child classes.
  cache_class = None

  #: The default coder used by the cache. If None, the coder is inferred.
  default_coder = None

  def get_writer_kwargs(self, data=None):
    """Additional arguments to pass through to the writer."""
    return {}

  def check_coder(self, write_fn, data):
    inferred_coder = self.default_coder or coders.registry.get_coder(
        datatype_inference.infer_element_type(data))
    cache = self.cache_class(self.location, **self.get_writer_kwargs(data))
    cache.requires_coder = True
    self.assertEqual(cache.coder, self.default_coder)
    write_fn(cache, data)
    self.assertEqual(cache.coder, inferred_coder)
    cache.truncate()
    self.assertEqual(cache.coder, self.default_coder)


class FileCoderTestBase(CoderTestBase):

  # Attributes to be set by child classes.
  cache_class = None
  location = None

  #: The default coder used by the cache. If None, the coder is inferred.
  default_coder = None

  def setUp(self):
    self._temp_dir = tempfile.mkdtemp()
    self.location = FileSystems.join(self._temp_dir, self.cache_class.__name__)

  def tearDown(self):
    FileSystems.delete([self._temp_dir])

  @parameterized.expand([
      (write_fn.__name__, write_fn, data)
      for data in GENERIC_TEST_DATA
      for write_fn in [write_through_pipeline, write_directly]
  ])
  def test_coder(self, _, write_fn, data):
    # TODO(ostrokach): Trying to mock out the `self.cache_class._writer_class`
    # attribute leads to a stack overflow error, but there must be a way.
    self.check_coder(write_fn, data)


class TextBasedCacheCoderTest(FileCoderTestBase, unittest.TestCase):

  cache_class = filebasedcache.TextBasedCache


class SafeTextBasedCacheCoderTest(FileCoderTestBase, unittest.TestCase):

  cache_class = filebasedcache.SafeTextBasedCache
  default_coder = filebasedcache.SafeFastPrimitivesCoder()


class TFRecordBasedCacheCoderTest(FileCoderTestBase, unittest.TestCase):

  cache_class = filebasedcache.TFRecordBasedCache


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
