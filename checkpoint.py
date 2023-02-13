import json
import numpy as np

DIM_FIELDS = {
    1: {'name': 'size'},
    2: {'name': 'name'},
}
SHAPE_FIELDS = {
    2: {'name': 'dim', 'fields': DIM_FIELDS, 'repeated': True},
    3: {'name': 'unknown_rank'},
}
ENTRY_FIELDS = {
    1: {'name': 'dtype'},
    2: {'name': 'shape', 'fields': SHAPE_FIELDS},
    3: {'name': 'shard_id'},
    4: {'name': 'offset'},
    5: {'name': 'size'},
    6: {'name': 'crc32c'},
    7: {'name': 'slices', 'repeated': True},
}


def read_varint(fh):
    bs = []
    while True:
        b = fh.read(1)[0]
        if b & 0b10000000:
            bs.append(b ^ 0b10000000)
        else:
            bs.append(b)
            break
    result = 0
    for i, b in enumerate(bs):
        result |= b << (i * 7)
    return result, len(bs)


def read_int(fh, size):
    result = 0
    for i, b in enumerate(fh.read(size)):
        result |= b << (i * 8)
    return result


def read_protobuf(fh, size, fields={}):
    result = {}
    while size > 0:
        tag = fh.read(1)[0]
        field = tag >> 3
        wire_type = tag & 0b111
        size -= 1

        field_data = fields.get(field, {})
        key = field_data.get('name', field)

        if wire_type == 0:
            value, k = read_varint(fh)
            size -= k
        elif wire_type == 1:
            value = read_int(fh, 8)
            size -= 8
        elif wire_type == 2:
            length, k = read_varint(fh)
            if 'fields' in field_data:
                value = read_protobuf(fh, length, field_data['fields'])
            else:
                value = fh.read(length)
            size -= k + length
        elif wire_type == 5:
            value = read_int(fh, 4)
            size -= 4
        else:
            raise NotImplementedError(wire_type)

        if field_data.get('repeated'):
            result.setdefault(key, [])
            result[key].append(value)
        else:
            result[key] = value

    assert size == 0
    return result


def get_index(path):
    result = {}
    with open(path / 'model.ckpt.index', 'rb') as fh:
        last_key = b''
        while True:
            shared, _ = read_varint(fh)
            non_shared, _ = read_varint(fh)
            value_size, _ = read_varint(fh)
            key = fh.read(non_shared)
            last_key = last_key[:shared] + key
            value = read_protobuf(fh, value_size, ENTRY_FIELDS)
            if last_key:
                result[last_key.decode('utf-8')] = value
            elif result:
                return result


class Model:
    def __init__(self, path):
        self.path = path
        self.index = get_index(path)
        with open(path / 'hparams.json') as fh:
            self.hparams = json.load(fh)

        self.data = {}
        with open(self.path / 'model.ckpt.data-00000-of-00001', 'rb') as fh:
            for key, entry in self.index.items():
                fh.seek(entry.get('offset', 0))
                dims = [d['size'] for d in entry['shape']['dim'] if d['size'] != 1]
                count = np.prod(dims)
                data = np.fromfile(fh, dtype='float32', count=count)
                data = data.reshape(dims)
                self.data[key] = data

    def get(self, key):
        return self.data[key]
