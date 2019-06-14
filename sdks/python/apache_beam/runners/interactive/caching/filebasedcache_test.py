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
import tempfile
import time
import unittest
import uuid

import mock
from parameterized import parameterized

from apache_beam import transforms
from apache_beam.io.filesystems import FileSystems
from apache_beam.runners.interactive.caching import filebasedcache
from apache_beam.runners.interactive.caching import filebasedcache_it_test
from apache_beam.runners.interactive.caching import pcollectioncache_it_test


class FileBasedCacheTest(unittest.TestCase):

  def _cache_class(self, location, *args, **kwargs):

    class MockedFileBasedCache(filebasedcache.FileBasedCache):

      _reader_class = mock.MagicMock()
      _writer_class = mock.MagicMock()
      _reader_passthrough_arguments = {}

    return MockedFileBasedCache(location, *args, **kwargs)

  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()
    self.location = FileSystems.join(self.temp_dir, self._cache_class.__name__)

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
    """Test that the constructor correctly validates arguments."""
    cache = self._cache_class(self.location)  # pylint: disable=unused-variable

    with self.assertRaises(ValueError):
      _ = self._cache_class(self.location, mode=None)

    with self.assertRaises(ValueError):
      _ = self._cache_class(self.location, mode="be happy")

  def test_overwrite_cache_0(self):
    cache = self._cache_class(self.location)  # pylint: disable=unused-variable
    # Refuse to create a cache with the same data storage location
    with self.assertRaises(IOError):
      _ = self._cache_class(self.location)

  def test_overwrite_cache_1(self):
    cache = self._cache_class(self.location)  # pylint: disable=unused-variable
    # Refuse to create a cache with the same data storage location
    _ = self.create_dummy_file(self.location)
    with self.assertRaises(IOError):
      _ = self._cache_class(self.location)

  def test_overwrite_cache_2(self):
    cache = self._cache_class(self.location)  # pylint: disable=unused-variable
    # OK to overwrite cache when in "overwrite" mode
    _ = self._cache_class(self.location, mode="overwrite")

  def test_overwrite_cache_3(self):
    cache = self._cache_class(self.location)
    # OK to overwrite cleared cache
    cache.remove()
    _ = self._cache_class(self.location)

  def test_timestamp(self):
    """Test that the timestamp increases with successive writes."""
    # Seems to always pass when delay=0.01 but set higher to prevent flakiness
    cache = self._cache_class(self.location)
    timestamp0 = cache.timestamp
    time.sleep(0.1)
    _ = self.create_dummy_file(self.location)
    timestamp1 = cache.timestamp
    self.assertGreater(timestamp1, timestamp0)

  def test_writer_arguments(self):
    """Test that the writer arguments get correctly passed onto the writer."""
    kwargs = {"a": 10, "b": "hello"}
    cache = self._cache_class(self.location, **kwargs)
    cache.writer()
    _, kwargs_out = list(cache._writer_class.call_args)
    self.assertEqual(kwargs_out, kwargs)

  def test_reader_arguments(self):
    """Test that the reader arguemnts get correctly passed onto the reader."""

    def check_reader_passthrough_kwargs(kwargs, passthrough):
      cache = self._cache_class(self.location, mode="overwrite", **kwargs)
      cache._reader_passthrough_arguments = passthrough
      cache.reader()
      _, kwargs_out = list(cache._reader_class.call_args)
      self.assertEqual(kwargs_out, {k: kwargs[k] for k in passthrough})

    check_reader_passthrough_kwargs({"a": 10, "b": "hello world"}, {})
    check_reader_passthrough_kwargs({"a": 10, "b": "hello world"}, {"b"})

  def test_writer(self):
    """Test that a new writer is constructed each time `writer()` is called."""
    cache = self._cache_class(self.location)
    self.assertEqual(cache._writer_class.call_count, 0)
    cache.writer()
    self.assertEqual(cache._writer_class.call_count, 1)
    cache.writer()
    self.assertEqual(cache._writer_class.call_count, 2)

  def test_reader(self):
    """Test that a new reader is constructed each time `reader()` is called."""
    cache = self._cache_class(self.location)
    self.assertEqual(cache._reader_class.call_count, 0)
    cache.reader()
    self.assertEqual(cache._reader_class.call_count, 1)
    cache.reader()
    self.assertEqual(cache._reader_class.call_count, 2)

  def test_write(self):
    """Test the implementation of `write()`."""
    cache = self._cache_class(self.location)
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
    """Test the implementation of `read()`."""
    cache = self._cache_class(self.location)
    _ = self.create_dummy_file(self.location)
    self.assertEqual(cache._reader_class()._source.read.call_count, 0)
    cache.read()
    # ..._source._read does not get called unless we get items from iterator
    self.assertEqual(cache._reader_class()._source.read.call_count, 0)
    with self.assertRaises(StopIteration):
      next(cache.read())
    self.assertEqual(cache._reader_class()._source.read.call_count, 1)
    list(cache.read())
    self.assertEqual(cache._reader_class()._source.read.call_count, 2)

  def test_truncate(self):
    """Test that `truncate()` correctly cleans up files."""
    cache = self._cache_class(self.location)
    self.assertEqual(
        len(list(filebasedcache.glob_files(cache.file_pattern))), 1)
    _ = self.create_dummy_file(self.location)
    self.assertEqual(
        len(list(filebasedcache.glob_files(cache.file_pattern))), 2)
    cache.truncate()
    self.assertEqual(
        len(list(filebasedcache.glob_files(cache.file_pattern))), 1)

  def test_clear_data(self):
    """Test that `clear()` correctly cleans up files."""
    cache = self._cache_class(self.location)
    self.assertEqual(
        len(list(filebasedcache.glob_files(cache.file_pattern))), 1)
    _ = self.create_dummy_file(self.location)
    self.assertEqual(
        len(list(filebasedcache.glob_files(cache.file_pattern))), 2)
    cache.remove()
    self.assertEqual(
        len(list(filebasedcache.glob_files(cache.file_pattern))), 0)

  def test_clear_metadata(self):
    """Test that `clear()` correctly cleans up metadata."""
    cache = self._cache_class(self.location, coder="mock")
    self.assertEqual(cache._writer_kwargs.get("coder"), "mock")
    cache.remove()
    self.assertEqual(cache._writer_kwargs.get("coder"), "mock")

    cache = self._cache_class(self.location)
    cache._writer_kwargs["coder"] = "mock"
    self.assertEqual(cache._writer_kwargs.get("coder"), "mock")
    cache.remove()
    self.assertTrue("coder" not in cache._writer_kwargs)

  @unittest.skip("Not implemented.")
  def test_garbage_collection(self):
    pass


class CoderTestBase(pcollectioncache_it_test.CoderTestBase):
  """Make sure that the coder gets set correctly when we write data to cache.

  Only applicable to implementations which infer the coder when writing data.
  """

  def setUp(self):
    self._temp_dir = tempfile.mkdtemp()
    self.location = FileSystems.join(self._temp_dir, self.cache_class.__name__)

  def tearDown(self):
    FileSystems.delete([self._temp_dir])

  @parameterized.expand([
      ("{}-{}".format(data_name, write_fn.__name__), write_fn, data)
      for data_name, data in pcollectioncache_it_test.GENERIC_TEST_DATA
      for write_fn in [
          filebasedcache_it_test.write_directly,
          filebasedcache_it_test.write_through_pipeline
      ]
  ])
  def test_coder(self, _, write_fn, data):
    dummy_ptransfrom = transforms.Map(lambda e: e)
    dummy_ptransfrom._sink = mock.MagicMock()
    with mock.patch("apache_beam.runners.interactive.caching.filebasedcache.{}"
                    "._writer_class".format(
                        self.cache_class.__name__)) as mock_writer:
      mock_writer.return_value = dummy_ptransfrom
      self.check_coder(write_fn, data)


class TextBasedCacheCoderTest(CoderTestBase, unittest.TestCase):

  cache_class = filebasedcache.TextBasedCache


class SafeTextBasedCacheCoderTest(CoderTestBase, unittest.TestCase):

  cache_class = filebasedcache.SafeTextBasedCache
  default_coder = filebasedcache.SafeFastPrimitivesCoder()


class TFRecordBasedCacheCoderTest(CoderTestBase, unittest.TestCase):

  cache_class = filebasedcache.TFRecordBasedCache


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
