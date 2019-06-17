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

import contextlib
import functools
import re
import time
import uuid

from future.moves import queue

from apache_beam import coders
from apache_beam.io.gcp.pubsub import PubsubMessage
from apache_beam.io.gcp.pubsub import ReadFromPubSub
from apache_beam.io.gcp.pubsub import WriteToPubSub
from apache_beam.runners.direct.direct_runner import _DirectWriteToPubSubFn
from apache_beam.runners.interactive.caching import PCollectionCache
from apache_beam.runners.interactive.caching.datatype_inference import infer_element_type
from apache_beam.transforms import Map
from apache_beam.transforms import PTransform
from apache_beam.utils.timestamp import Timestamp

try:
  from weakref import finalize
except ImportError:
  from apache_beam.runners.interactive.backports.weakref import finalize

try:
  from google.cloud import pubsub_v1
  from google.api_core import exceptions as gexc
except ImportError:
  pubsub = None

__all__ = [
    "StreamBasedCache",
    "PubSubBasedCache",
]


class StreamBasedCache(PCollectionCache):
  pass


class PubSubBasedCache(StreamBasedCache):

  _reader_class = ReadFromPubSub
  _writer_class = WriteToPubSub
  _reader_passthrough_arguments = {
      "id_label",
      "with_attributes",
      "timestamp_attribute",
      "coder",
  }

  def __init__(self,
               location,
               mode="error",
               coder=None,
               single_use=False,
               persist=False,
               **writer_kwargs):
    self.location = location
    self._single_use = single_use
    self._writer_kwargs = writer_kwargs
    self._timestamp = 0
    self._num_reads = 0
    self._coder_was_provided = "coder" in writer_kwargs
    self._finalizer = (lambda: None) if persist else finalize(
        self, finalize_pubsub_cache, self.location)

    _, project_id, _, topic_name = self.location.split('/')
    self.subscription_path = 'projects/{}/subscriptions/{}'.format(
        project_id, topic_name)
    self.snapshot_path = 'projects/{}/snapshots/{}'.format(
        project_id, topic_name) if not self._single_use else None

    if re.search("projects/.+/topics/.+", location) is None:
      raise ValueError(
          "'location' must be the path to a pubsub subscription in the form: "
          "'projects/{project}/topics/{topic}'.")
    if mode not in ['error', 'append', 'overwrite']:
      raise ValueError(
          "mode must be set to one of: ['error', 'append', 'overwrite'].")

    pub_client = pubsub_v1.PublisherClient()
    sub_client = pubsub_v1.SubscriberClient()

    try:
      _ = pub_client.create_topic(self.location)
    except gexc.AlreadyExists:
      if mode == "error":
        raise IOError("Topic '{}' already exists.".format(self.location))

    try:
      _ = sub_client.create_subscription(self.subscription_path, self.location)
    except gexc.AlreadyExists:
      if mode == "error":
        raise IOError("Subscription '{}' already exists.".format(
            self.subscription_path))
      if mode == "overwrite":
        sub_client.delete_subscription(self.subscription_path)
        _ = sub_client.create_subscription(self.subscription_path,
                                           self.location)

    if not self._single_use:
      try:
        _ = sub_client.create_snapshot(self.snapshot_path,
                                       self.subscription_path)
      except gexc.AlreadyExists:
        if mode == "error":
          raise IOError("Snapshot '{}' already exists.".format(
              self.snapshot_path))
        if mode == "overwrite":
          sub_client.delete_snapshot(self.snapshot_path)
          _ = sub_client.create_snapshot(self.snapshot_path,
                                         self.subscription_path)

  @property
  def timestamp(self):
    return self._timestamp

  def reader(self, from_start=True, **reader_kwargs):
    self._assert_topic_exists()
    self._assert_read_valid()
    self._num_reads += 1

    kwargs = {
        k: v
        for k, v in self._writer_kwargs.items()
        if k in self._reader_passthrough_arguments
    }
    kwargs.update(reader_kwargs)

    if "subscription" not in kwargs:
      if not self._single_use:
        kwargs["subscription"] = self._create_new_subscription(
            self.location, self.snapshot_path if from_start else None)
        # TODO(ostrokach): This subscription is currently not cleaned up
        # until cache.remove() is called.
      else:
        kwargs["subscription"] = self.subscription_path
    reader = PatchedReader(self._reader_class, (), kwargs)
    return reader

  def writer(self):
    self._timestamp = time.time()
    writer = PatchedWriter(self._writer_class, (self.location,),
                           self._writer_kwargs)
    return writer

  @contextlib.contextmanager
  def _read_to_queue(self, from_start=True, **reader_kwargs):
    self._assert_topic_exists()
    self._assert_read_valid()

    kwargs = {
        k: v
        for k, v in self._writer_kwargs.items()
        if k in self._reader_passthrough_arguments
    }
    kwargs.update(reader_kwargs)

    @functools.total_ordering
    class PQItem(tuple):

      def __lt__(self, other):
        return self[0].__lt__(other[0])

      def __eq__(self, other):
        return self[0].__eq__(other[0])

    # Set arbitrary queue size limit to prevent OOM errors.
    parsed_message_queue = queue.PriorityQueue(10000)

    def get_element(message, coder, with_attributes, timestamp_attribute):
      # TODO(ostrokach): Eventually, it would be nice to have a shared codebase
      # with `direct.transform_evaluator._PubSubReadEvaluator`.
      from datetime import datetime
      import pytz

      parsed_message = PubsubMessage._from_message(message)
      parsed_message.data = coder.decode(parsed_message.data)
      if (with_attributes and timestamp_attribute and
          timestamp_attribute in parsed_message.attributes):
        rfc3339_or_milli = parsed_message.attributes[timestamp_attribute]
        try:
          timestamp = Timestamp(micros=int(rfc3339_or_milli) * 1000)
        except ValueError:
          try:
            dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S%z')
          except ValueError as e:
            raise ValueError("Bad timestamp value: '{}'.".format(e))
          else:
            dt_delta = dt.astimezone(
                pytz.UTC) - datetime.utcfromtimestamp(0).replace(
                    tzinfo=pytz.UTC)
            timestamp = Timestamp(seconds=dt_delta.total_seconds())
      else:
        dt_delta = message.publish_time - datetime.utcfromtimestamp(0).replace(
            tzinfo=pytz.UTC)
        timestamp = Timestamp(seconds=dt_delta.total_seconds())

      if not with_attributes:
        parsed_message = parsed_message.data

      return timestamp, parsed_message

    def callback(message, kwargs=kwargs):
      message.ack()
      timestamp, parsed_message = get_element(
          message,
          coder=kwargs["coder"],
          with_attributes=kwargs.get("with_attributes"),
          timestamp_attribute=kwargs.get("timestamp_attribute"))
      pq_item = PQItem((timestamp, parsed_message))
      parsed_message_queue.put(pq_item)

    sub_client = pubsub_v1.SubscriberClient()
    subscription_path = self._get_or_create_subscription(
        kwargs.get("subscription"),
        snapshot_path=self.snapshot_path if from_start else None)
    future = sub_client.subscribe(subscription_path, callback=callback)
    try:
      yield parsed_message_queue
    finally:
      future.cancel()
      sub_client.delete_subscription(subscription_path)

  def read(self,
           from_start=True,
           return_timestamp=False,
           timeout=None,
           **reader_kwargs):

    def iter_parsed_messages(message_queue):
      while True:
        try:
          yield message_queue.get(timeout=timeout)
        except queue.Empty:
          return

    with self._read_to_queue(
        from_start=from_start, **reader_kwargs) as message_queue:
      for timestamp, parsed_message in iter_parsed_messages(message_queue):
        if return_timestamp:
          yield timestamp, parsed_message
        else:
          yield parsed_message

  def write(self, elements):
    if self._infer_coder:
      # TODO(ostrokach): We might want to infer the element type from the first
      # N elements, rather than reading the entire iterator.
      elements = list(elements)
      element_type = infer_element_type(elements)
      coder = coders.registry.get_coder(element_type)
      self._writer_kwargs["coder"] = coder

    writer_kwargs = self._writer_kwargs.copy()
    coder = writer_kwargs.pop("coder")
    _ = writer_kwargs.pop("timestamp_attribute", None)
    writer = self._writer_class(self.location, **writer_kwargs)

    do_fn = _DirectWriteToPubSubFn(writer._sink)
    do_fn.start_bundle()
    try:
      for element in elements:
        element_bytes = coder.encode(element)
        if do_fn.with_attributes:
          element_bytes = PubsubMessage(
              element_bytes,
              {a: str(element[a]) for a in do_fn.with_attributes})
        do_fn.process(element_bytes)
    finally:
      do_fn.finish_bundle()

  def truncate(self):
    sub_client = pubsub_v1.SubscriberClient()
    try:
      sub_client.delete_snapshot(self.snapshot_path)
      sub_client.delete_subscription(self.subscription_path)
    except gexc.NotFound:
      pass
    _ = sub_client.create_subscription(self.subscription_path, self.location)
    _ = sub_client.create_snapshot(self.snapshot_path, self.subscription_path)

  def remove(self):
    self._finalizer()
    if not self._coder_was_provided and "coder" in self._writer_kwargs:
      del self._writer_kwargs["coder"]

  @property
  def removed(self):
    return not self._finalizer.alive

  def __del__(self):
    self.remove()

  @property
  def _infer_coder(self):
    return (not self._writer_kwargs.get("coder") and
            "coder" in self._reader_passthrough_arguments)

  def _assert_topic_exists(self):
    pub_client = pubsub_v1.PublisherClient()
    try:
      _ = pub_client.get_topic(self.location)
    except gexc.NotFound:
      raise IOError("Pubsub topic '{}' does not exist.".format(self.location))

  def _assert_read_valid(self):
    if self._single_use and self._num_reads >= 1:
      raise ValueError(
          "A single-use cache allows only a single read over its lifetime.")

  def _get_or_create_subscription(self,
                                  subscription_path=None,
                                  snapshot_path=None):
    if subscription_path:
      return subscription_path
    elif self._single_use:
      return self.subscription_path
    else:
      subscription_path = self._create_new_subscription(self.location,
                                                        snapshot_path)
      return subscription_path

  def _create_new_subscription(self,
                               topic_path,
                               snapshot_path=None,
                               suffix=None):
    if suffix is None:
      suffix = "-{}".format(uuid.uuid4().hex)
    sub_client = pubsub_v1.SubscriberClient()
    subscription_path = (
        topic_path.replace("/topics/", "/subscriptions/") + suffix)
    _ = sub_client.create_subscription(subscription_path, topic_path)
    if snapshot_path is not None:
      sub_client.seek(subscription_path, snapshot=snapshot_path)
    return subscription_path


def finalize_pubsub_cache(location):
  return remove_topic_and_descendants(location)


class PatchedReader(PTransform):

  def __init__(self, reader_class, reader_args, reader_kwargs):
    self._reader_class = reader_class
    self._reader_args = reader_args
    self._reader_kwargs = reader_kwargs

  def expand(self, pbegin):
    reader_kwargs = self._reader_kwargs.copy()
    coder = reader_kwargs.pop("coder")
    with_attributes = reader_kwargs.get("with_attributes")

    def decode_element(element):
      if with_attributes:
        element.data = coder.decode(element.data)
      else:
        element = coder.decode(element)
      return element

    reader = self._reader_class(*self._reader_args, **reader_kwargs)
    return pbegin | reader | Map(decode_element)


class PatchedWriter(PTransform):
  """

  .. note::
    This function updates the 'writer_kwargs' dictionary by assigning to
    the 'coder' key an instance of the inferred coder.
  """

  def __init__(self, writer_class, writer_args, writer_kwargs):
    self._writer_class = writer_class
    self._writer_args = writer_args
    self._writer_kwargs = writer_kwargs

  @property
  def _writer(self):
    writer_kwargs = {
        k: v for k, v in self._writer_kwargs.items() if k not in ["coder"]
    }
    writer = self._writer_class(*self._writer_args, **writer_kwargs)
    return writer

  def expand(self, pcoll):
    if "coder" not in self._writer_kwargs:
      coder = coders.registry.get_coder(pcoll.element_type)
      self._writer_kwargs["coder"] = coder

    writer_kwargs = self._writer_kwargs.copy()
    coder = writer_kwargs.pop("coder")
    with_attributes = writer_kwargs.get("with_attributes")
    timestamp_attribute = writer_kwargs.pop("timestamp_attribute", None)
    if (timestamp_attribute is not None and
        timestamp_attribute not in with_attributes):
      raise ValueError(
          "timestamp_attribute must be included in with_attributes.")

    def encode_element(element):

      if with_attributes:
        element = PubsubMessage(
            data=coder.encode(element),
            attributes={a: str(element[a]) for a in with_attributes})
      else:
        element = coder.encode(element)

      return element

    writer = self._writer_class(*self._writer_args, **writer_kwargs)
    return pcoll | Map(encode_element) | writer


def remove_topic_and_descendants(topic_path):
  project_path = "/".join(topic_path.split('/')[:2])
  pub_client = pubsub_v1.PublisherClient()
  try:
    _ = pub_client.get_topic(topic_path)
  except gexc.NotFound:
    return
  sub_client = pubsub_v1.SubscriberClient()
  try:
    for snapshot in sub_client.list_snapshots(project_path):
      if snapshot.topic == topic_path:
        sub_client.delete_snapshot(snapshot.name)
  except gexc.MethodNotImplemented:
    pass
  for subscription in pub_client.list_topic_subscriptions(topic_path):
    sub_client.delete_subscription(subscription)
  pub_client.delete_topic(topic_path)
