from crewai import Crew, Process
from agents import context_manager, specialist, safety_shield
from tasks import validation_task, diagnosis_task, safety_audit_task, final_output_task

# 에이전트 조립
medical_crew = Crew(
    agents=[context_manager, specialist, safety_shield],
    tasks=[validation_task, diagnosis_task, safety_audit_task, final_output_task],
    process=Process.sequential, # 순차적으로 진행
    verbose=True
)

if __name__ == "__main__":
    print("에이전트 가동")
    
    #userdata
    user_inputs = {
        "user_data": "최근 2주간 볼 쪽에 붉은기 확산, 가려움 동반. 수면 부족(하루 4시간), 기름진 음식 섭취 잦음.",
        "image_analysis": "Erythema(홍반) 감지, 경계가 불분명함, 멜라닌 수치 정상."
    }

    result = medical_crew.kickoff(inputs=user_inputs)
    
    print("\n\n" + "="*50)
    print("FINAL MEDICAL REPORT")
    print("="*50)
    print(result)