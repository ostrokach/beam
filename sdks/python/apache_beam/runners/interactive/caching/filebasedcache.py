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

import urllib
import warnings

from apache_beam import coders
from apache_beam.io import avroio
from apache_beam.io import parquetio
from apache_beam.io import textio
from apache_beam.io import tfrecordio
from apache_beam.io.filesystems import FileSystems
from apache_beam.runners.interactive.caching import pcollectioncache
from apache_beam.testing import datatype_inference
from apache_beam.transforms import PTransform

try:
  from weakref import finalize
except ImportError:
  from backports.weakref import finalize

try:  # Python 3
  unquote_to_bytes = urllib.parse.unquote_to_bytes
  quote = urllib.parse.quote
except AttributeError:  # Python 2
  # pylint: disable=deprecated-urllib-function
  unquote_to_bytes = urllib.unquote
  quote = urllib.quote

__all__ = [
    "FileBasedCache",
    "TextBasedCache",
    "SafeTextBasedCache",
    "TFRecordBasedCache",
    "ParquetBasedCache",
    "AvroBasedCache",
    "SafeFastPrimitivesCoder",
]


class FileBasedCache(pcollectioncache.PCollectionCache):

  def __init__(self, location, mode="error", persist=False, **writer_kwargs):
    self.location = location
    self._writer_kwargs = writer_kwargs
    self.coder = writer_kwargs.pop("coder", None)
    self._coder_was_provided = self.coder is not None
    self._num_writes = 0
    self._persist = persist
    self._finalizer = (
        finalize(self, lambda: None) if self._persist else finalize(
            self, lambda pattern: FileSystems.delete(
                list(glob_files(pattern + "**"))), self.file_pattern))

    # TODO(ostrokach): Implement append mode.
    if mode not in ['error', 'overwrite']:
      raise ValueError("'mode' must be set to one of: ['error', 'overwrite'].")
    exitsting_files = list(glob_files(self.file_pattern))
    if mode == "error" and exitsting_files:
      raise IOError("The following cache files already exist: {}.".format(
          exitsting_files))
    if mode == "overwrite":
      self.truncate()

    root, _ = FileSystems.split(self._file_path_prefix)
    try:
      FileSystems.mkdirs(root)
    except IOError:
      pass
    # It is possible to read from am empty stream, so it should also be possible
    # to read from an empty file.
    FileSystems.create(self._file_path_prefix + ".empty").close()

  @property
  def persist(self):
    return self._persist

  @persist.setter
  def persist(self, persist):
    self._persist = persist
    self._finalizer = (lambda: None) if self._persist else finalize(
        self, lambda pattern: FileSystems.delete(list(glob_files(pattern))),
        self.file_pattern)

  @property
  def timestamp(self):
    timestamp = 0
    for path in glob_files(self.file_pattern):
      timestamp = max(timestamp, FileSystems.last_updated(path))
    return timestamp

  def reader(self, **kwargs):
    reader_kwargs = self._reader_kwargs.copy()
    if self.requires_coder:
      reader_kwargs["coder"] = self.coder
    reader_kwargs.update(kwargs)
    reader = self._reader_class(self.file_pattern, **reader_kwargs)
    # Keep a reference to the parent object so that cache does not get garbage
    # collected while the pipeline is running.
    reader._cache = self
    return reader

  def writer(self):
    self._num_writes += 1
    if self.requires_coder and self.coder is None:
      writer = PatchedWriter(self, self._writer_class,
                             (self._file_path_prefix,), self._writer_kwargs)
      return writer

    if self.requires_coder:
      self._writer_kwargs["coder"] = self.coder
    writer = self._writer_class(self._file_path_prefix, **self._writer_kwargs)
    # Keep a reference to the parent object so that cache does not get garbage
    # collected while the pipeline is running.
    writer._cache = self
    return writer

  def read(self, **kwargs):
    reader_kwargs = self._reader_kwargs.copy()
    if self.requires_coder:
      reader_kwargs["coder"] = self.coder
    reader_kwargs.update(kwargs)
    source = self.reader(**reader_kwargs)._source
    range_tracker = source.get_range_tracker(None, None)
    for element in source.read(range_tracker):
      yield element

  def write(self, elements):
    self._num_writes += 1
    if self.requires_coder and self.coder is None:
      # TODO(ostrokach): We might want to infer the element type from the first
      # N elements, rather than reading the entire iterator.
      elements = list(elements)
      element_type = datatype_inference.infer_element_type(elements)
      self.coder = coders.registry.get_coder(element_type)
    sink = self.writer()._sink
    handle = sink.open(self._file_path_prefix)
    try:
      for element in elements:
        sink.write_record(handle, element)
    finally:
      sink.close(handle)

  def truncate(self):
    FileSystems.delete(list(glob_files(self.file_pattern)))
    FileSystems.create(self._file_path_prefix + ".empty").close()
    if not self._coder_was_provided and self.coder is not None:
      self.coder = None

  def remove(self):
    self._finalizer()

  @property
  def removed(self):
    return not self._finalizer.alive

  def __del__(self):
    self.remove()

  @property
  def file_pattern(self):
    return self.location + '*'

  @property
  def _reader_kwargs(self):
    reader_kwargs = {
        k: v
        for k, v in self._writer_kwargs.items()
        if k in self._reader_passthrough_arguments
    }
    return reader_kwargs

  @property
  def _file_path_prefix(self):
    return self.location + "-{:03d}".format(self._num_writes)


class TextBasedCache(FileBasedCache):

  _reader_class = textio.ReadFromText
  _writer_class = textio.WriteToText
  _reader_passthrough_arguments = {"coder", "compression_type"}
  requires_coder = True

  def __init__(self, location, **writer_kwargs):
    warnings.warn("TextBasedCache is not reliable and should not be used.")
    super(TextBasedCache, self).__init__(location, **writer_kwargs)


class SafeTextBasedCache(FileBasedCache):

  _reader_class = textio.ReadFromText
  _writer_class = textio.WriteToText
  _reader_passthrough_arguments = {"coder", "compression_type"}
  requires_coder = True

  def __init__(self, location, **writer_kwargs):
    writer_kwargs["coder"] = SafeFastPrimitivesCoder()
    super(SafeTextBasedCache, self).__init__(location, **writer_kwargs)


class TFRecordBasedCache(FileBasedCache):

  _reader_class = tfrecordio.ReadFromTFRecord
  _writer_class = tfrecordio.WriteToTFRecord
  _reader_passthrough_arguments = {"coder", "compression_type"}
  requires_coder = True


class ParquetBasedCache(FileBasedCache):

  _reader_class = parquetio.ReadFromParquet
  _writer_class = parquetio.WriteToParquet
  _reader_passthrough_arguments = {}
  requires_coder = False


class AvroBasedCache(FileBasedCache):

  _reader_class = avroio.ReadFromAvro
  _writer_class = avroio.WriteToAvro
  _reader_passthrough_arguments = {"use_fastavro"}
  requires_coder = False


class PatchedWriter(PTransform):

  def __init__(self, cache, writer_class, writer_args, writer_kwargs):
    self._cache = cache
    self._writer_class = writer_class
    self._writer_args = writer_args
    self._writer_kwargs = writer_kwargs

  def expand(self, pcoll):
    if self._cache.requires_coder and self._cache.coder is None:
      self._cache.coder = coders.registry.get_coder(pcoll.element_type)
    writer_kwargs = self._writer_kwargs.copy()
    if self._cache.requires_coder:
      writer_kwargs["coder"] = self._cache.coder
    writer = self._writer_class(*self._writer_args, **writer_kwargs)
    return pcoll | writer


class SafeFastPrimitivesCoder(coders.Coder):
  """This class add an quote/unquote step to escape special characters."""

  # pylint: disable=deprecated-urllib-function

  def encode(self, value):
    return quote(
        coders.coders.FastPrimitivesCoder().encode(value)).encode('utf-8')

  def decode(self, value):
    return coders.coders.FastPrimitivesCoder().decode(unquote_to_bytes(value))


def glob_files(pattern):
  match = FileSystems.match([pattern])
  assert len(match) == 1
  for metadata in match[0].metadata_list:
    yield metadata.path
