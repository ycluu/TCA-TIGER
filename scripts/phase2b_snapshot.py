"""Capture real Phase-1 losses before trainer integration changes."""
import json
import torch
from scripts.phase2a_preference import setup, CP, OUT
from scripts.audit_tiger_generation import load_model
from genrec.trainers.tiger_trainer import CollateFn
from genrec.trainers.tca_loss import select_training_loss

if __name__ == '__main__':
    path = OUT/'phase2b_before.pt'
    if path.exists():
        raise FileExistsError(path)
    ds,tca,_ = setup()
    ids = json.loads((OUT/'scale_audit.json').read_text())['rows'][0]['indices']
    b = CollateFn(256,3,3,150)([ds[i] for i in ids])
    m = load_model(CP,'cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {'indices':ids}
    with torch.no_grad():
        for amp in [False,True]:
            with torch.amp.autocast('cuda',enabled=amp,dtype=torch.float16):
                loss,z = m(b['input_ids'].cuda(),b['attention_mask'].cuda(),b['labels'].cuda())
            result[str(amp)] = float(select_training_loss(loss,objective='tca_full_vocab',logits=z,
                labels=b['labels'].cuda(),teacher_probs=tca['cf_token_probs'][ids].cuda().float(),alpha=.1,temperature=1))
    torch.save(result,path)
    print(result)
