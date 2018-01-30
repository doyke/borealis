# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: driverpacket.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='driverpacket.proto',
  package='driverpacket',
  syntax='proto3',
  serialized_pb=_b('\n\x12\x64riverpacket.proto\x12\x0c\x64riverpacket\"\xb7\x02\n\x0c\x44riverPacket\x12\x10\n\x08\x63hannels\x18\x01 \x03(\r\x12\x41\n\x0f\x63hannel_samples\x18\x02 \x03(\x0b\x32(.driverpacket.DriverPacket.SamplesBuffer\x12\x14\n\x0csequence_num\x18\x04 \x01(\r\x12\x0e\n\x06txrate\x18\x05 \x01(\x01\x12\x14\n\x0ctxcenterfreq\x18\x03 \x01(\x01\x12\x14\n\x0crxcenterfreq\x18\x06 \x01(\x01\x12\x1e\n\x16numberofreceivesamples\x18\n \x01(\r\x12\x19\n\x11timetosendsamples\x18\x07 \x01(\x01\x12\x0b\n\x03SOB\x18\x08 \x01(\x08\x12\x0b\n\x03\x45OB\x18\t \x01(\x08\x1a+\n\rSamplesBuffer\x12\x0c\n\x04real\x18\x01 \x03(\x01\x12\x0c\n\x04imag\x18\x02 \x03(\x01\x62\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_DRIVERPACKET_SAMPLESBUFFER = _descriptor.Descriptor(
  name='SamplesBuffer',
  full_name='driverpacket.DriverPacket.SamplesBuffer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='real', full_name='driverpacket.DriverPacket.SamplesBuffer.real', index=0,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='imag', full_name='driverpacket.DriverPacket.SamplesBuffer.imag', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=305,
  serialized_end=348,
)

_DRIVERPACKET = _descriptor.Descriptor(
  name='DriverPacket',
  full_name='driverpacket.DriverPacket',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='channels', full_name='driverpacket.DriverPacket.channels', index=0,
      number=1, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='channel_samples', full_name='driverpacket.DriverPacket.channel_samples', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='sequence_num', full_name='driverpacket.DriverPacket.sequence_num', index=2,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='txrate', full_name='driverpacket.DriverPacket.txrate', index=3,
      number=5, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='txcenterfreq', full_name='driverpacket.DriverPacket.txcenterfreq', index=4,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rxcenterfreq', full_name='driverpacket.DriverPacket.rxcenterfreq', index=5,
      number=6, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='numberofreceivesamples', full_name='driverpacket.DriverPacket.numberofreceivesamples', index=6,
      number=10, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='timetosendsamples', full_name='driverpacket.DriverPacket.timetosendsamples', index=7,
      number=7, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='SOB', full_name='driverpacket.DriverPacket.SOB', index=8,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='EOB', full_name='driverpacket.DriverPacket.EOB', index=9,
      number=9, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_DRIVERPACKET_SAMPLESBUFFER, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=37,
  serialized_end=348,
)

_DRIVERPACKET_SAMPLESBUFFER.containing_type = _DRIVERPACKET
_DRIVERPACKET.fields_by_name['channel_samples'].message_type = _DRIVERPACKET_SAMPLESBUFFER
DESCRIPTOR.message_types_by_name['DriverPacket'] = _DRIVERPACKET

DriverPacket = _reflection.GeneratedProtocolMessageType('DriverPacket', (_message.Message,), dict(

  SamplesBuffer = _reflection.GeneratedProtocolMessageType('SamplesBuffer', (_message.Message,), dict(
    DESCRIPTOR = _DRIVERPACKET_SAMPLESBUFFER,
    __module__ = 'driverpacket_pb2'
    # @@protoc_insertion_point(class_scope:driverpacket.DriverPacket.SamplesBuffer)
    ))
  ,
  DESCRIPTOR = _DRIVERPACKET,
  __module__ = 'driverpacket_pb2'
  # @@protoc_insertion_point(class_scope:driverpacket.DriverPacket)
  ))
_sym_db.RegisterMessage(DriverPacket)
_sym_db.RegisterMessage(DriverPacket.SamplesBuffer)


# @@protoc_insertion_point(module_scope)
