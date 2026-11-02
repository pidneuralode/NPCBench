"""Keep CPU image/tensor queues bounded while continuously feeding vLLM."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,threading,time

def vllm_batch(model,batch,emit=None,progress_path=None):
    window,chunk=48,8
    values=[None]*len(batch);indices={};traces={};pending=set();cursor=0
    started=time.time();halt=threading.Event()
    progress=dict(started_at_unix=started,batch_size=len(batch),host_inflight_limit=window,
                  submitted_requests=0,live_requests=0,engine_steps=0,finished_requests=0,status='preparing')
    def persist():
        if progress_path:
            tmp=progress_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(progress,checked_at_unix=time.time(),elapsed_seconds=time.time()-started)))
            tmp.replace(progress_path)
    def heartbeat():
        while not halt.wait(15):
            persist();print('NPCBench bounded generation '+json.dumps(progress),flush=True)
    monitor=threading.Thread(target=heartbeat,daemon=True);monitor.start()
    try:
        with ThreadPoolExecutor(max_workers=chunk) as pool:
            while cursor<len(batch) or pending:
                if cursor<len(batch) and len(pending)<window:
                    count=min(chunk,window-len(pending),len(batch)-cursor)
                    items=batch[cursor:cursor+count]
                    prepared=list(pool.map(lambda x:model.process_messages(x[1]),items))
                    begin=model.llm.request_counter.counter
                    for offset,((record,actual),value) in enumerate(zip(items,prepared)):
                        images=value.get('multi_modal_data',{}).get('image',[])
                        if len(images)!=len(actual['images']):raise ValueError('Missing delivered image')
                        rid=str(begin+offset);indices[rid]=cursor+offset;pending.add(rid)
                        traces[rid]=dict(delivered_images=len(images),image_sizes=[list(x.size) for x in images],
                            rendered_prompt_sha256=hashlib.sha256(value['prompt'].encode()).hexdigest())
                    ids=model.llm.enqueue(prepared,sampling_params=model.sampling_params,use_tqdm=False)
                    assert len(ids)==count and model.llm.request_counter.counter==begin+count
                    cursor+=count;progress.update(submitted_requests=cursor,live_requests=len(pending))
                    # vLLM owns submitted tensors; do not retain decoded PIL batches here.
                    del prepared,items,images,value
                if model.llm.llm_engine.has_unfinished_requests():
                    progress['status']='generating'
                    outputs=model.llm.llm_engine.step();progress['engine_steps']+=1
                    progress['last_step_at_unix']=time.time()
                    for result in outputs:
                        if not result.finished:continue
                        rid=result.request_id;i=indices[rid]
                        assert rid in pending and values[i] is None
                        c=result.outputs[0];elapsed=time.time()-started
                        value=dict(response=c.text,finish_reason=c.finish_reason,stop_reason=c.stop_reason,
                            output_tokens=len(c.token_ids),prompt_tokens=len(result.prompt_token_ids or []),
                            completion_token_ids=list(c.token_ids),media_trace=traces.pop(rid),elapsed_seconds=elapsed,
                            timing_scope='completion since bounded batch start',batch_elapsed_seconds=elapsed,
                            batch_size=len(batch),host_inflight_limit=window,submission_chunk=chunk,
                            batch_sample_ids=[r['sample_id'] for r,_ in batch])
                        values[i]=value;pending.remove(rid);progress['finished_requests']+=1
                        if emit:emit(batch[i][0],value)
                    progress['live_requests']=len(pending)
                elif pending:raise ValueError('Engine ended before all submitted requests returned')
        assert all(v is not None for v in values)
        progress['status']='completed'
    finally:
        halt.set();monitor.join(timeout=2);persist()
    return values
