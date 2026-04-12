import os
from crewai import Agent, LLM
from tools import medical_search_tool
from dotenv import load_dotenv

load_dotenv()

# 고지능 엔진 (전문의, 비평가용 - Groq)
groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# 보안 엔진 (매니저용 - 로컬 Ollama)
ollama_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434"
)

# 1. 컨텍스트 매니저 (개인정보 보호 및 공감, Ollama)
context_manager = Agent(
    role='컨텍스트 매니저 및 게이트키퍼',
    goal='사용자 데이터의 누락을 체크하고, 최종 결과를 따뜻한 말투로 보정하여 전달합니다.',
    backstory='''당신은 환자의 첫인상과 끝인상을 책임집니다. 데이터가 부족하면 분석을 중단시키고
    추가 입력을 요청해야 합니다. 말투는 적당한 공감이 섞인 친근한 존댓말을 사용하세요.''',
    llm=ollama_llm, # 수정된 Ollama LLM 객체 연결
    verbose=True,
    allow_delegation=True
)

# 2. 임상 전문의 (Groq)
specialist = Agent(
    role='임상 전문의 (SOAP 분석가)',
    goal='이미지 분석 결과와 문진 데이터를 기반으로 보수적인 감별 진단을 수행합니다.',
    backstory='''당신은 15년 경력의 피부과 전문의입니다. 위음성 리스크를 줄이기 위해 
    미세한 가능성도 놓치지 않고 SOAP 형식에 맞춰 진단합니다.''',
    llm=groq_llm,
    tools=[medical_search_tool],
    allow_delegation=False
)

# 3. 안전 비평가 (근거 검증, Groq, 추후 효율성 고도화)
safety_shield = Agent(
    role='의학 통계 안전 비평가',
    goal='전문의의 판단을 논문 근거로 검증하고, 크롬바흐 알파 0.8 미만인 연구는 배제합니다.',
    backstory='''당신은 매우 깐깐한 검토관입니다. 연구의 신뢰도 계수가 0.8 이상인지, 
    이해관계가 얽혀있는지 확인하여 위음성 리스크가 30%를 넘는지 판정합니다.''',
    llm=groq_llm,
    allow_delegation=False
)