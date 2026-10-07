import ast
import subprocess
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/serve_live_research_dashboard.py'


def refresh_source():
    tree = ast.parse(SOURCE.read_text())
    html = next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PAGE' for t in n.targets))
    return html.split('let refreshInFlight=false;', 1)[1].split("for(const id of ['flow-model'", 1)[0]


def test_refresh_deduplicates_and_recovers_after_success_and_failure():
    program = '''
let refreshInFlight=false, calls=0, resolveFetch;
const elements={}; const $=id=>(elements[id]??={});
let fetch=()=>{calls++;return new Promise(resolve=>{resolveFetch=resolve;});};
const renderLead=()=>{}, renderAdmission=()=>{}, renderCurrentFlow=()=>{}, render=()=>{}, renderActiveExperiment=()=>{};
''' + refresh_source() + '''
(async()=>{
 const first=refresh(); await refresh(); await refresh();
 if(calls!==1)throw Error('overlapping fetches');
 resolveFetch({ok:true,json:async()=>({})});await first;
 if(refreshInFlight)throw Error('success lock retained');
 fetch=async()=>{calls++;throw Error('network fixture');};
 await refresh(); if(refreshInFlight)throw Error('failure lock retained');
 if(!elements['clock'].textContent.includes('network fixture'))throw Error('failure not shown');
 fetch=async()=>{calls++;return {ok:true,json:async()=>({})};};
 await refresh(); if(calls!==3||refreshInFlight)throw Error('retry did not recover');
 fetch=async()=>{calls++;return {ok:true,json:async()=>{throw Error('bad JSON');}};};
 await refresh();if(refreshInFlight)throw Error('JSON failure lock retained');
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    result = subprocess.run(['node'], input=program, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
