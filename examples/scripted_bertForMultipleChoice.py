import torch
import io, contextlib, tempfile
from pytorch_pretrained_bert.modeling import BertForMultipleChoice, BertConfig
from contextlib import contextmanager
from torch._C import _jit_python_print


#@contextmanager  # noqa: T484
#def TemporaryFileName():
#    with tempfile.NamedTemporaryFile() as f:
#        yield f.name
#
#
#@contextlib.contextmanager
#def freeze_rng_state():
#    rng_state = torch.get_rng_state()
#    if torch.cuda.is_available():
#        cuda_rng_state = torch.cuda.get_rng_state()
#    yield
#    if torch.cuda.is_available():
#        torch.cuda.set_rng_state(cuda_rng_state)
#    torch.set_rng_state(rng_state)
#
#
#def getExportImportCopy(m, also_test_file=True, map_location=None):
#    if isinstance(m, torch._C.Function):
#        src, constants = _jit_python_print(m)
#        cu = torch.jit.CompilationUnit()._import(src, constants)
#        return getattr(cu, m.name)
#
#    buffer = io.BytesIO()
#    torch.jit.save(m, buffer)
#    buffer.seek(0)
#    imported = torch.jit.load(buffer, map_location=map_location)
#
#    if not also_test_file:
#        return imported
#
#    with TemporaryFileName() as fname:
#        imported.save(fname)
#        return torch.jit.load(fname, map_location=map_location)
#
#
#def assertExportImportModule(m, inputs):
#    m_import = getExportImportCopy(m)
#    res1 = runAndSaveRNG(m, inputs),
#    res2 = runAndSaveRNG(m_import, inputs)
#    print(res1)
#    print(res2)
#
#
#def runAndSaveRNG(func, inputs, kwargs=None):
#    kwargs = kwargs if kwargs else {}
#    with freeze_rng_state():
#        results = func(*inputs, **kwargs)
#    return results
torch.manual_seed(666)

input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

num_choices = 2

model = BertForMultipleChoice(config, num_choices)
res = model(input_ids, token_type_ids, input_mask)
print(res)
# logits = model(input_ids, token_type_ids, input_mask)
# assertExportImportModule(model, (input_ids, token_type_ids, input_mask))
