"""Standalone provider/model checker and smoke tester.
Examples:
  python run_models.py --list
  python run_models.py --list --provider groq
  python run_models.py --provider groq --model allam-2-7b
"""
import os, argparse, asyncio
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError: pass

async def list_models(provider):
    if provider == 'groq':
        from groq import AsyncGroq
        c=AsyncGroq(api_key=os.environ['GROQ_API_KEY']); r=await c.models.list(); return [x.id for x in r.data]
    if provider == 'mistral':
        try: from mistralai.client import Mistral
        except ImportError: from mistralai import Mistral
        c=Mistral(api_key=os.environ['MISTRAL_API_KEY']); r=c.models.list(); return [getattr(x,'id',str(x)) for x in getattr(r,'data',r)]
    if provider == 'gemini':
        from google import genai
        c=genai.Client(api_key=os.environ['GEMINI_API_KEY']); return [getattr(x,'name','').replace('models/','') for x in c.models.list()]
    raise ValueError(provider)

async def smoke(provider, model, prompt):
    if provider=='groq':
        from groq import AsyncGroq
        c=AsyncGroq(api_key=os.environ['GROQ_API_KEY']); kw={}
        if 'gpt-oss' in model: kw={'reasoning_effort':'low','reasoning_format':'parsed'}
        r=await c.chat.completions.create(model=model,messages=[{'role':'user','content':prompt}],temperature=0,max_tokens=100,**kw); return r.choices[0].message.content or ''
    if provider=='mistral':
        try: from mistralai.client import Mistral
        except ImportError: from mistralai import Mistral
        c=Mistral(api_key=os.environ['MISTRAL_API_KEY']); r=await c.chat.complete_async(model=model,messages=[{'role':'user','content':prompt}],temperature=0,max_tokens=100); return r.choices[0].message.content or ''
    if provider=='gemini':
        from google import genai
        c=genai.Client(api_key=os.environ['GEMINI_API_KEY'])
        r=await asyncio.to_thread(lambda:c.models.generate_content(model=model,contents=prompt,config={'temperature':0,'max_output_tokens':100})); return getattr(r,'text','') or ''
    raise ValueError(provider)

async def main():
    p=argparse.ArgumentParser(); p.add_argument('--list',action='store_true'); p.add_argument('--provider',choices=['groq','mistral','gemini']); p.add_argument('--model'); p.add_argument('--prompt',default='What is 17 + 25? Answer with just the number.'); a=p.parse_args()
    if a.list:
        ps=[a.provider] if a.provider else ['groq','mistral','gemini']
        for x in ps:
            try:
                print('\n'+x.upper()); print('\n'.join(sorted(await list_models(x))))
            except Exception as e: print(f'[{x}] ERROR: {e}')
    elif a.model:
        if not a.provider: p.error('--provider is required with --model')
        print(await smoke(a.provider,a.model,a.prompt))
    else: p.print_help()
asyncio.run(main())
