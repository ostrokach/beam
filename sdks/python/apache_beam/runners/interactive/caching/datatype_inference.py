import array
import json

import pyarrow as pa
from past.builtins import unicode

from apache_beam.typehints import typehints
from apache_beam.typehints import trivial_inference

import numpy as np


def infer_element_type(elements):
  element_type = typehints.Union[[
      trivial_inference.instance_to_type(e) for e in elements
  ]]
  return element_type


def infer_column_coders(data):
  column_data = {}
  for row in data:
    for key, value in row.items():
      column_data.setdefault(key, []).append(value)
  column_coders = {
      key:
      typehints.Union[[trivial_inference.instance_to_type(v) for v in value]]
      for key, value in column_data.items()
  }
  return column_coders


def infer_avro_schema(data, use_fastavro=False):
  _typehint_to_avro_type = {
      typehints.Union[[int]]: "int",
      typehints.Union[[int, None]]: ["int", "null"],
      typehints.Union[[long]]: "long",
      typehints.Union[[long, None]]: ["long", "null"],
      typehints.Union[[float]]: "double",
      typehints.Union[[float, None]]: ["double", "null"],
      typehints.Union[[str]]: "string",
      typehints.Union[[str, None]]: ["string", "null"],
      typehints.Union[[unicode]]: "string",
      typehints.Union[[unicode, None]]: ["string", "null"],
      typehints.Union[[np.ndarray]]: "bytes",
      typehints.Union[[np.ndarray, None]]: ["bytes", "null"],
      typehints.Union[[array.array]]: "bytes",
      typehints.Union[[array.array, None]]: ["bytes", "null"],
  }

  column_coders = infer_column_coders(data)
  avro_fields = [{
      "name": str(key),
      "type": _typehint_to_avro_type[value]
  } for key, value in column_coders.items()]
  schema_dict = {
      "namespace": "example.avro",
      "type": "record",
      "name": "User",
      "fields": avro_fields
  }
  if use_fastavro:
    from fastavro import parse_schema
    return parse_schema(schema_dict)
  else:
    import avro.schema
    return avro.schema.parse(json.dumps(schema_dict))


def infer_parquet_schema(data):
  column_data = {}
  for row in data:
    for key, value in row.items():
      column_data.setdefault(key, []).append(value)
  column_types = {
      key: pa.array(value).type for key, value in column_data.items()
  }
  return pa.schema(list(column_types.items()))
