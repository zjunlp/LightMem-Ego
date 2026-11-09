"""Offline regressions for failed multi-period and prefix retrieval requests."""
from types import SimpleNamespace
import ast
from pathlib import Path
import threading

import pytest

from online_query_constraints import parse_query_constraints, constraint_window, constraint_windows, overlaps
from online_query.memory_router import MemoryRouter
from online_query.memory_fusion import MemoryFusion
from online_query.evidence_validation import validate_evidence
from online_short_term.mst_retriever import MSTRetriever


@pytest.mark.parametrize('question, expected', [
 ('请分别给第0秒和第120秒的画面，比较她头上的物品与脸部穿戴', [[0,0],[120,120]]),
 ('请比较第420秒与第470秒的上衣、口罩、眼镜和头上物品', [[420,420],[470,470]]),
 ('分别在第60～67秒、第195～207秒、第233～240秒、第429～436秒找手机', [[60,67],[195,207],[233,240],[429,436]]),
 ('请分别给出第0～180秒、第330～430秒、第450～707秒的穿戴状态，概括变化', [[0,180],[330,430],[450,707]]),
 ('第0～120秒和第343～353秒内拿手机，比较口罩状态', [[0,120],[343,353]]),
 ('截至第450秒，拿手机是否出现在穿外套之前和之后', [[0,450]]),
 ('视频前两分钟内拿手机', [[0,120]]),
 ('最近三分钟里有没有戴墨镜', [[508,688]]),
 ('第10～20秒在做什么', [[10,20]]),
 ('第2分钟之后', [[120,688]]),
])
def test_real_failed_questions_preserve_all_constraints(question, expected):
    actual = parse_query_constraints(question, 688)
    assert actual['time_ranges'] == expected
    assert actual['time_range'] == (expected[0] if len(expected)==1 else None)


def test_union_does_not_admit_gap_and_incomplete_expression_is_not_first_point():
    c = parse_query_constraints('第0～120秒和第343～353秒', 707)
    assert not overlaps(200,250,constraint_window(c))
    assert overlaps(350,350,constraint_window(c))
    c = parse_query_constraints('比较第1小时和第30秒', 707)
    assert c['time_parse_incomplete'] and not c['time_explicit'] and not constraint_windows(c)
    assert not parse_query_constraints('找两段相隔至少3秒的手机动作',45)['time_explicit']


def test_router_keeps_old_history_and_current_side_for_two_points():
    state={'current_ready':True,'current_stale':False,'short_term_ready':True,'long_term_ready':True,
           'mcur_time_span':[91,120], 'mst_time_span':[0,112], 'mlt_time_span':[0,90]}
    d=MemoryRouter().route('请分别给第0秒和第120秒的画面，比较穿戴',{},state,{})
    assert d['memory_route']['use_current'] and d['memory_route']['use_long_term']


def test_short_term_search_reserves_late_window_despite_repeated_high_ranked_early_events():
    events=[{'event_id':str(i),'start_time':i,'end_time':i+1,'event_caption_refined':'phone holding phone white shirt'} for i in range(8)]
    events += [{'event_id':'gap','start_time':200,'end_time':210,'event_caption_refined':'phone phone'},
               {'event_id':'late','start_time':350,'end_time':353,'event_caption_refined':'holding phone black coat'}]
    store=SimpleNamespace(get_state=lambda:{'mst_version':1},load_events=lambda:events)
    retriever=MSTRetriever(store)
    c=parse_query_constraints('第0～10秒和第343～353秒拿手机',707)
    selected=retriever.search('拿手机',top_k=2,cache_context={'query_constraints':c})
    assert len(selected)==2 and 'late' in [r['event_id'] for r in selected]
    assert 'gap' not in [r['event_id'] for r in selected]


def test_fusion_preserves_every_window_under_small_budget():
    c=parse_query_constraints('第0～10秒和第343～353秒',707)
    events=[{'memory_id':str(i),'start_time':i,'end_time':i+1,'score':1.,'caption':'phone'} for i in range(6)]
    events += [{'memory_id':'late','start_time':343,'end_time':353,'score':.1,'caption':'phone'},
               {'memory_id':'gap','start_time':200,'end_time':250,'score':1.,'caption':'phone'}]
    fused=MemoryFusion().fuse({'text_results':events},{'query_constraints':c,'memory_priority':{'M_lt':1}},
        {'final_evidence_k':2,'retrieval_plan':{'M_lt':{'candidate_budget':2}}},'temporal_localization')
    chosen=fused['final_evidence']
    assert len(chosen)==2 and any(e['evidence_id']=='late' for e in chosen)
    assert not any(e['evidence_id']=='gap' for e in chosen)


def test_validation_reports_missing_side_and_gap_even_with_two_events():
    c=parse_query_constraints('第0秒和第120秒',120)
    v=validate_evidence('比较', [{'start_time':0,'end_time':30},{'start_time':90,'end_time':120}],
        [{'capture_time_seconds':0},{'capture_time_seconds':60}],c)
    assert v['uncovered_requested_ranges']==[[120,120]]
    assert 'sent_image_outside_requested_time' in v['issues']
    assert 'requested_instant_has_no_direct_image' in v['issues']


def test_dense_retrieval_executes_union_filter_and_reserves_low_score_side_without_api():
    # Execute the actual dense retrieval method with an offline, precomputed
    # three-document index. Avoid loading production model packages entirely.
    import numpy as np
    path=Path('src/em2mem/memory/episodic/EpisodicMemory_rag.py')
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(x,ast.FunctionDef) and x.name=='set_query_constraints' for x in n.body))
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'retrieve_ranked','_matches_requested_time'}]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),ast.ClassDef(name='Offline',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(module)
    namespace={'np':np,'logger':SimpleNamespace(info=lambda *a:None,warning=lambda *a:None)};exec(compile(module,str(path),'exec'),namespace)
    obj=namespace['Offline']();entries=[SimpleNamespace(doc_id='early',start_time=0.,end_time=10.),SimpleNamespace(doc_id='gap',start_time=200.,end_time=210.),SimpleNamespace(doc_id='late',start_time=343.,end_time=353.)]
    obj._thread_local=threading.local();obj._thread_local.query_constraints=parse_query_constraints('第0～10秒和第343～353秒',707)
    obj.dense_index_built=True;obj.granularities=['30sec'];obj.doc_embeddings={'30sec':np.array([[1.,0.],[1.,0.],[.1,0.]])}
    obj._get_thread_indexed_time=lambda:707;obj._get_query_embedding=lambda q:np.array([1.,0.]);obj._get_thread_prefix_size=lambda g:3
    obj._lookup_entry_from_index=lambda g,i:entries[i];obj._graph_aware_rerank=lambda **kw:kw['candidates'];obj._expand_temporal_neighbors=lambda *a:[];obj._expand_entity_neighbors=lambda *a:[]
    selected=obj.retrieve_ranked('phone',{'30sec':2})
    assert [e.doc_id for e,_ in selected]==['early','late']


def test_shared_units_composite_clocks_and_numeric_before_after():
    assert parse_query_constraints('比较第420和470秒',707)['time_ranges']==[[420,420],[470,470]]
    assert parse_query_constraints('比较第1分30秒和第2分30秒',707)['time_ranges']==[[90,90],[150,150]]
    assert parse_query_constraints('比较00:30和01:30',707)['time_ranges']==[[30,30],[90,90]]
    assert parse_query_constraints('第30秒之前和之后的手机',100)['time_ranges']==[[0,30],[30,100]]


def test_graph_pool_filter_keeps_two_sides_and_rejects_gap_without_embedding():
    path=Path('src/em2mem/memory/episodic/EpisodicMemory_rag.py');tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(x,ast.FunctionDef) and x.name=='set_query_constraints' for x in n.body))
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'retrieve_ranked_from_doc_id_pool','_matches_requested_time'}]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),ast.ClassDef(name='Offline',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(module)
    namespace={'_tokenize':lambda q:set(q.split())};exec(compile(module,str(path),'exec'),namespace)
    obj=namespace['Offline']();obj._thread_local=threading.local();obj._thread_local.query_constraints=parse_query_constraints('第0～10秒和第343～353秒',707)
    entries={str(i):SimpleNamespace(doc_id=str(i),start_time=i,end_time=i+1,timestamp_int=[i,i+1]) for i in range(6)}
    entries.update(gap=SimpleNamespace(doc_id='gap',start_time=200,end_time=210,timestamp_int=[200,210]),late=SimpleNamespace(doc_id='late',start_time=343,end_time=353,timestamp_int=[343,353]))
    obj.dense_index_built=True;obj.doc_embeddings={};obj._family_policy=lambda info:{'question_family':'event'};obj._get_thread_indexed_time=lambda:707
    obj.normalize_doc_ids_to_30s=lambda ids:ids;obj.expand_seed_doc_ids_with_neighbors=lambda ids,**kw:list(entries);obj.get_caption_by_doc_id=lambda d,g:entries.get(d)
    obj._family_direct_score=lambda e,*a: .1 if e.doc_id=='late' else 1.;obj._projection_local_score_30s=lambda *a:0.
    chosen=obj.retrieve_ranked_from_doc_id_pool('phone',list(entries),final_top_k=2,max_candidates=3)
    assert 'late' in [e.doc_id for e,_ in chosen] and 'gap' not in [e.doc_id for e,_ in chosen]


def test_repeated_action_comparison_keeps_two_observed_supplements_within_budget():
    rows=[{'memory_id':'neighbor','start_time':360,'end_time':390,'score':1.,'caption':'putting on coat'},
          {'memory_id':'before','start_time':60,'end_time':90,'score':.1,'caption':'holding phone','supplemental':True},
          {'memory_id':'after','start_time':420,'end_time':450,'score':.1,'caption':'holding phone again','supplemental':True}]
    c=parse_query_constraints('拿手机是在穿外套之前还是之后，请分别给证据',688)
    chosen=MemoryFusion().fuse({'text_results':rows},{'query_constraints':c,'memory_priority':{'M_lt':1}},
          {'final_evidence_k':2,'retrieval_plan':{'M_lt':{'candidate_budget':4}}},'temporal_reasoning')['final_evidence']
    assert {r['evidence_id'] for r in chosen}=={'before','after'}


@pytest.mark.parametrize('question', ['请比较第0、120、240秒三张图','请比较第0、120与240秒','Compare 0, 120 and 240 seconds'])
def test_shared_unit_three_points_never_drops_first(question):
    assert parse_query_constraints(question,300)['time_ranges']==[[0,0],[120,120],[240,240]]


@pytest.mark.parametrize('question', [
    '按时间概括这六分钟里前方车辆的变化，区分早期、中间和末段，给关键画面。',
    '概括这三分钟内的活动变化', '总结这段180秒里的穿戴变化',
    'Summarize this 6 minutes of video', 'Give a summary of this 180 seconds of footage',
])
def test_clip_duration_summary_is_not_future_point(question):
    c=parse_query_constraints(question,179)
    assert not c['time_explicit'] and c['time_ranges']==[] and c['scope']=='history'


def test_negative_clause_does_not_add_required_endpoint_or_expand_positive_range():
    c=parse_query_constraints('只在第0～90秒内找图；不要使用第90秒以后的画面',90)
    assert c['time_ranges']==[[0,90]]
    assert not overlaps(91,91,constraint_window(c))
    c=parse_query_constraints('Find frames in 0 to 120 seconds; do not use frames after 90 seconds',120)
    assert c['time_ranges']==[[0,90]]
    c=parse_query_constraints('只看第0～100秒，不要使用第20～30秒画面',100)
    assert overlaps(10,10,constraint_window(c)) and overlaps(31,31,constraint_window(c))
    assert not overlaps(20,20,constraint_window(c)) and not overlaps(25,25,constraint_window(c)) and not overlaps(30,30,constraint_window(c))
    c=parse_query_constraints('只看第20秒，不要使用第20秒画面',100)
    assert c['time_empty'] and not overlaps(20,20,constraint_window(c))
