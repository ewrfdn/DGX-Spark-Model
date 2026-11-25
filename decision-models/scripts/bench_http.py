import json, time, urllib.request, sys
PORT=sys.argv[1] if len(sys.argv)>1 else "8014"
PARA=("Customer reported that the checkout flow failed after rotating the API key and that "
      "the duplicate charge is still pending on the corporate card. ")
def payload(n_tok, questions=2):
    state=(PARA*(n_tok*4//len(PARA)+1))[:n_tok*4]
    q={"tier":{"type":"choice","instructions":"Which tier?","criteria":{"small":"x","mid":"y","frontier":"z"}},
       "escalate":{"type":"noul","instructions":"Needs frontier or human?"}}
    if questions>=5:
        q.update({"topic":{"type":"choice","instructions":"Topic?","criteria":{"billing":"m","tech":"b","sales":"s"}},
                  "urgent":{"type":"noul","instructions":"Urgent?"},
                  "mood":{"type":"score","instructions":"Mood?","criteria":["calm","annoyed","angry"]}})
    return {"model":"x","messages":[{"role":"user","content":json.dumps({"state":state,"questions":q})}]}
def call(body):
    req=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    t=time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r: d=json.load(r)
    wall=(time.perf_counter()-t)*1000
    return wall, d.get("usage",{})
print(f"port={PORT}")
for n in (1000, 4000, 8000, 16000, 24000):
    for rep in range(2):
        w,u=call(payload(n))
        print(f"{n:6d} tok  rep{rep}  wall={w:8.0f} ms  server={u.get('latency_ms')}  in={u.get('input_tokens')} seq={u.get('sequences')}", flush=True)
