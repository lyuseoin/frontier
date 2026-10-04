"""Privacy boundary. No logging, disk storage, provider SDK or network imports."""
import hashlib
import json
import math
import re
import time
import unicodedata

class SecurityError(ValueError):
    pass

RULES = [
    ('이메일', r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}'),
    ('전화번호', r'(?<!\d)(?:\+82[- .]?|0)1[016789][- .]?\d{3,4}[- .]?\d{4}(?!\d)'),
    ('주민번호 형태', r'(?<!\d)\d{6}[- ]?[1-4]\d{6}(?!\d)'),
    ('웹 주소', r'(?:https?://|www\.)\S+'),
    ('계정명', r'(?<!\w)@[\w.\-]+'),
    ('인증키 형태', r'\b(?:AIza[\w-]{20,}|sk-[\w-]{16,})\b'),
]
INJECTION = re.compile(r'ignore.{0,30}(?:instruction|previous|system)|(?:이전|시스템|위의).{0,20}(?:무시|명령|지시)|system\s*:|developer\s*:|reveal.{0,20}(?:secret|key)|<\s*script', re.I)

def clean(text):
    if not isinstance(text,str) or len(text)>6000:raise SecurityError('입력은 6,000자 이내로 작성해 주세요.')
    return ''.join(c for c in unicodedata.normalize('NFKC',text) if c in '\n\t' or unicodedata.category(c)[0]!='C')

def redact(text):
    result=clean(text); counts={}
    for name,pat in RULES:
        result,n=re.subn(pat,'[가림]',result)
        if n:counts[name]=n
    return result,counts

def titles(text):
    rows=[x.strip() for x in clean(text).splitlines() if x.strip()]
    if not 3<=len(rows)<=30:raise SecurityError('제목을 한 줄에 하나씩 3~30개 입력해 주세요.')
    if any(len(x)>200 for x in rows):raise SecurityError('각 제목은 200자 이내로 줄여 주세요.')
    return rows

def fingerprint(rows,mode,model=''):
    return hashlib.sha256(json.dumps([rows,mode,model],ensure_ascii=False).encode()).hexdigest()

def payload(rows):
    return json.dumps({'titles':[{'id':i,'title':x} for i,x in enumerate(rows)]},ensure_ascii=False)

def validate_response(raw,rows,topics):
    if not isinstance(raw,str) or len(raw)>16000:raise SecurityError('AI 응답의 크기가 올바르지 않습니다.')
    try:data=json.loads(raw)
    except (ValueError,TypeError):raise SecurityError('AI 응답 형식이 올바르지 않습니다.') from None
    if not isinstance(data,list) or len(data)!=len(rows):raise SecurityError('AI 응답 항목 수가 일치하지 않습니다.')
    seen=set();result={}
    for item in data:
        if not isinstance(item,dict) or set(item)!={'id','topic','stance'}:raise SecurityError('허용하지 않은 응답 필드입니다.')
        idx=item['id'];v=item['stance']
        if type(idx) is not int or idx in seen or not 0<=idx<len(rows):raise SecurityError('응답 번호가 올바르지 않습니다.')
        if not isinstance(item['topic'],str) or item['topic'] not in topics:raise SecurityError('허용하지 않은 주제입니다.')
        if type(v) not in (int,float) or not math.isfinite(v) or not -1<=v<=1:raise SecurityError('관점 점수 범위를 벗어났습니다.')
        seen.add(idx);result[idx]={'label':rows[idx],'topic':item['topic'],'stance':float(v)}
    return [result[i] for i in range(len(rows))]

def authorize(rows,mode,model,consent_digest):
    # Check again at the network boundary, not only in the UI.
    titles('\n'.join(rows))
    if mode!='Google Gemini 분석' or consent_digest!=fingerprint(rows,mode,model):raise SecurityError('현재 전송 내용에 대한 동의가 필요합니다.')
    for row in rows:
        _,found=redact(row)
        if found:raise SecurityError('개인정보 형태가 남아 있습니다. 가린 뒤 다시 확인해 주세요.')
        if INJECTION.search(row):raise SecurityError('지시문 형태의 입력은 외부 전송할 수 없습니다. 해당 줄을 수정해 주세요.')

def reserve_request(state,now=None):
    now=time.monotonic() if now is None else now
    recent=[x for x in state.get('_api_times',[]) if now-x<3600]
    if recent and now-recent[-1]<15:raise SecurityError('연속 요청은 15초 간격으로 시도해 주세요.')
    if len(recent)>=10:raise SecurityError('이 세션의 시간당 10회 요청 한도에 도달했습니다.')
    state['_api_times']=recent+[now]

def clear_private(state):
    for k in list(state):
        if k!='_api_times':del state[k]

LOCAL_WORDS={
 '정치':['정치','선거','국회','대통령','민주','공화'], '경제':['경제','주식','투자','금리','부동산','금융'],
 '기술':['기술','인공지능','ai','it','코딩','보안','해킹','로봇'], '교육':['교육','공부','입시','수능','학교','학습'],
 '환경':['환경','기후','탄소','재활용','에너지'], '사회':['사회','뉴스','노동','정책','복지'],
 '건강':['건강','운동','스포츠','축구','수면','식단'], '문화':['문화','게임','영화','음악','아이돌','드라마'],
}
def local_classify(rows):
    items=[];unknown=[]
    for title in rows:
        low=title.lower();scores={t:sum(w in low for w in words) for t,words in LOCAL_WORDS.items()};best=max(scores,key=scores.get)
        if scores[best]==0 or list(scores.values()).count(scores[best])>1:unknown.append(title)
        else:items.append({'label':title,'topic':best,'stance':0.0})
    return items,unknown
