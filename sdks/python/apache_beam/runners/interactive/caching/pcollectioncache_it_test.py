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

import itertools
import logging
import sys

import numpy as np
from past.builtins import unicode

from apache_beam import coders
from apache_beam.runners.interactive.caching import datatype_inference

# yapf: disable
GENERIC_TEST_DATA = [
    #
    # ("empty_0", []),
    ("none_0", [None]),
    ("none_1", [None, None, None]),
    ("strings_0", ["ABC"]),
    ("strings_1", ["ABC", "DeF"]),
    ("strings_2", [u"ABC", u"±♠Ωℑ"]),
    ("numbers_0", [1.5]),
    ("numbers_1", [100, -123.456, 78.9]),
    ("numbers_2", [b"abc123"]),
    ("bytes_0", [b"abc123", b"aAaAa"]),
    ("bytes_1", ["ABC", 1.2, 100, 0, -10, None, b"abc123"]),
    ("mixed_primitive_types_0", [("a", "b", "c")]),
    ("tuples_0", [("a", 1, 1.2), ("b", 2, 5.5)]),
    ("tuples_1", [("a", 1, 1.2), (2.5, "c", None)]),
    ("tuples_2", [{"col1": "a", "col2": 1, "col3": 1.5}]),
    ("dictionaries_0", [{}]),
    ("dictionaries_1", [
        {"col1": "a", "col2": 1, "col3": 1.5},
        {"col1": "b", "col2": 2, "col3": 4.5}]),
    ("dictionaries_2", [
        {"col1": "a", "col2": 1, "col3": 1.5},
        {4: 1, 5: 3.4, (6, 7): "a"}]),
    ("dictionaries_3", [
        {"col1": "a", "col2": 1.5},
        {"col1": 1, "col3": u"±♠Ω"},
    ]),
    ("dictionaries_4", [{"a": 10}]),
    ("mixed_compound_types_0", [("a", "b", "c"), ["d", 1], {
        "col1": 1,
        202: 1.234
    }, None, "abc", b"def", 100, (1, 2, 3, "b")]),
    ("array_0", [np.zeros((3, 6)), np.ones((10, 22))])
]
# yapf: enable

# yapf: disable
DATAFRAME_TEST_DATA = [
    ("empty_0", []),
    ("empty_1", [{}]),
    ("empty_2", [{}, {}, {}]),
    ("missing_columns_0", [{"col1": "abc", "col2": "def"}, {"col1": "hello"}]),
    ("string_0", [
        {"col1": "abc", "col2": "def"},
        {"col1": "hello", "col2": "good bye"}]),
    ("string_1", [
        {"col1": b"abc", "col2": "def"},
        {"col1": b"hello", "col2": "good bye"}]),
    ("string_2", [
        {"col1": u"abc", "col2": u"±♠Ω"},
        {"col1": u"hello", "col2": u"Ωℑ"}]),
    ("numeric_0", [{"x": 123, "y": 5.55}, {"x": 555, "y": 6.63}]),
    ("numeric_1", [{"x": 123, "y": 5.55}, {"x": 555, "y": 6.63}]),
    ("array_0", [{"x": np.array([1, 2])}, {"x": np.array([3, 4, 5])}]),
]
# yapf: enable


class ExtraAssertions(object):

  if sys.version_info[0] < 3:

    def assertCountEqual(self, first, second, msg=None):
      """Assert that two containers have the same number of the same items in
      any order.
      """
      return self.assertItemsEqual(first, second, msg=msg)

  def assertArrayCountEqual(self, data1, data2):
    """Assert that two containers have the same items, with special treatment
    for numpy arrays.
    """
    # 'self.assertCountEqual' is much faster, so try it first.
    try:
      self.assertCountEqual(data1, data2)
      return
    except ValueError:
      pass

    try:
      iter(data1)
    except TypeError:
      self.assertEqual(data1, data2)
      return

    self.assertEqual(len(data1), len(data2))

    if isinstance(data1, (str, bytes, unicode)):
      self.assertEqual(data1, data2)
      return

    if isinstance(data1, dict):
      self.assertCountEqual(list(data1.keys()), list(data2.keys()))
      for key in data1:
        self.assertArrayCountEqual(data1[key], data2[key])
      return

    if isinstance(data1, np.ndarray):
      np.testing.assert_array_almost_equal(data1, data2)
      return

    # Performance here is terrible: O(n!), and larger for nested lists.
    for data2_perm in itertools.permutations(data2):
      try:
        for d1, d2 in zip(data1, data2_perm):
          self.assertArrayCountEqual(d1, d2)
      except AssertionError:
        continue
      return
    raise AssertionError(
        "The two objects '{}' and '{}' do not contain the same elements.".
        format(data1, data2))


class CoderTestBase(ExtraAssertions):

  # Attributes to be set by child classes.
  cache_class = None
  location = None

  #: The default coder used by the cache. If None, the coder is inferred.
  default_coder = None

  def get_writer_kwargs(self, data=None):
    """Additional arguments to pass through to the writer."""
    return {}

  def check_coder(self, write_fn, data):
    inferred_coder = self.default_coder or coders.registry.get_coder(
        datatype_inference.infer_element_type(data))
    cache = self.cache_class(self.location, **self.get_writer_kwargs(data))
    self.assertEqual(cache._writer_kwargs.get("coder"), self.default_coder)
    write_fn(cache, data)
    self.assertEqual(cache._writer_kwargs.get("coder"), inferred_coder)
    cache.remove()
    self.assertEqual(cache._writer_kwargs.get("coder"), self.default_coder)


class SerializationTestBase(ExtraAssertions):

  # Attributes to be set by child classes.
  cache_class = None
  location = None

  def get_writer_kwargs(self, data=None):
    return {}

  test_data = [{"a": 11, "b": "XXX"}, {"a": 20, "b": "YYY"}]

  def check_serde_empty(self, write_fn, read_fn, serializer):
    cache = self.cache_class(self.location,
                             **self.get_writer_kwargs(self.test_data))
    cache_out = serializer.loads(serializer.dumps(cache))
    write_fn(cache_out, self.test_data)
    data_out = list(read_fn(cache_out, limit=len(self.test_data)))
    self.assertEqual(data_out, self.test_data)

  def check_serde_filled(self, write_fn, read_fn, serializer):
    cache = self.cache_class(self.location,
                             **self.get_writer_kwargs(self.test_data))
    write_fn(cache, self.test_data)
    cache_out = serializer.loads(serializer.dumps(cache))
    data_out = list(read_fn(cache_out, limit=len(self.test_data)))
    self.assertEqual(data_out, self.test_data)


class RoundtripTestBase(ExtraAssertions):

  # Attributes to be set by child classes.
  cache_class = None
  location = None

  def get_writer_kwargs(self, data=None):
    return {}

  def check_roundtrip(self, write_fn, read_fn, data):
    """Make sure that data can be correctly written using the write_fn function
    and read using the read_fn function.
    """
    cache = self.cache_class(self.location, **self.get_writer_kwargs(data))
    write_fn(cache, data)
    data_out = read_fn(cache, limit=len(data))
    self.assertArrayCountEqual(data_out, data)
    write_fn(cache, data)
    data_out = read_fn(cache, limit=len(data) * 2)
    self.assertArrayCountEqual(data_out, data * 2)
    cache.truncate()
    data_out = read_fn(cache, timeout=1)
    self.assertArrayCountEqual(data_out, [])


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
