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

import queue
import sys
import threading

__all__ = ["create_data_publisher_app", "ServerThread"]


def create_data_publisher_app(data_source, processors=None, timeout=1):
  """Create a flask app that can serve data from the data_source.

  Args:
      data_source (queue.Queue): A source of data conforming to the Queue API.
      processors (List[callable]): A list of functions that will be applied to
          the data before serving.
      timeout (float): Time (in seconds) that we should wait when obtaining
          an element from the queue.

  .. warning::
      data_source should probably be thread-safe, or strange things might
      happen.
  """
  import flask
  app = flask.Flask(__name__)

  def crossdomain(f):
    """Allow access to the endpoint from different ports and domains."""

    def wrapped_function(*args, **kwargs):
      resp = flask.make_response(f(*args, **kwargs))
      h = resp.headers
      h["Access-Control-Allow-Origin"] = "*"
      h["Access-Control-Allow-Methods"] = "GET, OPTIONS, POST"
      h["Access-Control-Max-Age"] = str(21600)
      requested_headers = flask.request.headers.get(
          "Access-Control-Request-Headers")
      if requested_headers:
        h["Access-Control-Allow-Headers"] = requested_headers

      return resp

    return wrapped_function

  # TODO(ostrokach): Maybe should randomize the URL instead of using data,
  # in case anyone ever exposes this to the internet.
  @app.route("/data", methods=["GET", "OPTIONS", "POST"])
  @crossdomain
  def data():  # pylint: disable=unused-variable
    data = []
    try:
      data = [data_source.get(timeout=timeout)]
    except queue.Empty:
      pass
    else:
      if processors:
        for processor in processors:
          data = list(processor(data))
    return flask.jsonify(data)

  return app


class ServerThread(threading.Thread):
  """Serve the WSGI application app."""

  # Credit: https://stackoverflow.com/a/45017691/2063031

  def __init__(self, app, host="localhost", port=0, threaded=False):
    from werkzeug.serving import make_server
    super(ServerThread, self).__init__()
    self.server = make_server(host=host, port=port, app=app, threaded=threaded)
    self.context = app.app_context()
    self.context.push()
    if sys.version_info > (3, 4):
      self._finalizer = weakref.finalize(self, lambda server: server.shutdown(),
                                         self.server)
    else:
      self._finalizer = self.server.shutdown

  def run(self):
    self.server.serve_forever()

  def stop(self):
    self._finalizer()

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, *args):
    self.stop()


# class BokehThread(threading.Thread):

#   def __init__(self,
#                data_source,
#                data_sink,
#                plot_handle,
#                processors=None,
#                timeout=1,
#                rollover=None):
#     self._data_source = data_source
#     self._plot_handle = plot_handle
#     self._processors = processors
#     self._timeout = timeout
#     self._rollover = None
#     self._data_sink = data_sink

#   def run(self):
#     while True:
#       element = self._data_source.pull(timeout=self._timeout)
#       if self._processors is not None:
#         for processor in self._processors:
#           element = processor(element)
#         self._data_sink.stream(element, rollover=self._rollover)
