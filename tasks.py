#각 에이전트 별 태스크 정의
from crewai import Task
from agents import context_manager, specialist, safety_shield

# Task 1: 데이터 무결성 검사
validation_task = Task(
    description='''입력된 사용자 데이터(수면, 식습관, 증상 등) 중 빠진 것이 있는지 확인하세요.
    데이터가 부족하다면 "더 정확한 진단을 내릴 수 있어요."라는 문구와 함께 입력을 유도하세요.''',
    expected_output='데이터 완결성 보고서 또는 부족 데이터 보충 요청 메시지.',
    agent=context_manager
)

# Task 2: 임상 진단
diagnosis_task = Task(
    description='''전달받은 문진 데이터와 이미지 분석값을 바탕으로 감별 진단을 내리세요.
    최소 2개 이상의 가능성 있는 질환을 확률과 함께 제시하세요.''',
    expected_output='SOAP 형식의 임상 진단서 (질환명 및 확률 포함).',
    agent=specialist,
    context=[validation_task]
)

# Task 3: 안전성 및 신뢰도 검증
safety_audit_task = Task(
    description='''전문의의 진단 근거가 되는 연구의 Cronbach's Alpha 계수가 0.8 이상인지 확인하세요.
    위음성 리스크를 계산하여 30% 이상일 경우 '재진단 권고' 플래그를 설정하세요.''',
    expected_output='신뢰도 검증 수치 및 위음성 리스크 판정 결과.',
    agent=safety_shield,
    context=[diagnosis_task]
)

# Task 4: 최종 퍼스널라이징 및 출력
final_output_task = Task(
    description='''앞선 모든 결과를 종합하여 사용자에게 최종 메시지를 보냅니다.
    1. 1, 2순위 확률 차이가 10%p 이내면 둘 다 출력.
    2. 재진단 알림(Notification) 동의 여부 확인 문구 포함.
    3. 말투는 공감 위주의 친근한 존댓말.''',
    expected_output='사용자에게 전달될 최종 상담 리포트.',
    agent=context_manager,
    context=[safety_audit_task]
)