import requests
from bs4 import BeautifulSoup
from crewai.tools import tool  # crewai only, remove langchain import

@tool("medical_search_tool")
def medical_search_tool(url: str) -> str:
    """MSD 매뉴얼 등 의학 가이드라인의 내용을 읽어옵니다."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # ← bug fix from earlier review
        headers = {"User-Agent": "Mozilla/5.0"}  # ← bug fix from earlier review
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.find('section', class_='topic__content')
        return content.get_text() if content else "본문 내용을 찾을 수 없습니다."
    except Exception as e:
        return f"데이터 추출 중 오류 발생: {str(e)}"