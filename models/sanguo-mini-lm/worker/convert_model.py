import json, struct, os

SRC = '/tmp/opencode/mini_lm/sanguo.model.json'
OUT_DIR = '/tmp/opencode/sanguo-worker/weights'
os.makedirs(OUT_DIR, exist_ok=True)

m = json.load(open(SRC))
cfg = m['config']
entries = []
blob = bytearray()

def add(name, flat_values):
    off = len(blob) // 4
    n = len(flat_values)
    blob.extend(struct.pack('<%df' % n, *flat_values))
    entries.append({'name': name, 'offset': off, 'size': n})

def flat(mat):
    if isinstance(mat[0], list):
        return [x for row in mat for x in row]
    return list(mat)

add('embedding', flat(m['embedding']))
add('outputWeight', flat(m['outputWeight']))
add('outputBias', flat(m['outputBias']))
for i, b in enumerate(m['blocks']):
    a = b['attention']
    add('b%d.q' % i, flat(a['queryWeight']))
    add('b%d.k' % i, flat(a['keyWeight']))
    add('b%d.v' % i, flat(a['valueWeight']))
    add('b%d.o' % i, flat(a['outputWeight']))
    add('b%d.n1s' % i, flat(b['attentionNorm']['scale']))
    add('b%d.n1b' % i, flat(b['attentionNorm']['bias']))
    f = b['feedForward']
    add('b%d.w1' % i, flat(f['weight1']))
    add('b%d.b1' % i, flat(f['bias1']))
    add('b%d.w2' % i, flat(f['weight2']))
    add('b%d.b2' % i, flat(f['bias2']))
    add('b%d.n2s' % i, flat(b['feedForwardNorm']['scale']))
    add('b%d.n2b' % i, flat(b['feedForwardNorm']['bias']))

manifest = {
    'config': cfg,
    'tokenizer': m['tokenizer'],
    'vocabulary': m['vocabulary'],
    'entries': entries,
}

json_path = os.path.join(OUT_DIR, 'manifest.json')
bin_path = os.path.join(OUT_DIR, 'weights.bin')
json.dump(manifest, open(json_path, 'w'), ensure_ascii=False)
open(bin_path, 'wb').write(bytes(blob))
print('manifest ok:', os.path.getsize(json_path), 'bytes')
print('weights ok:', os.path.getsize(bin_path), 'bytes, entries:', len(entries))
print('total floats:', len(blob) // 4)
