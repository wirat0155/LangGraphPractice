"""
API สำหรับรายงานสภาพอากาศรายจังหวัด (ประเทศไทย)
แปลงจาก Gradio Interface เป็น REST API ที่เรียกผ่าน Postman ได้

การใช้งาน:
1. รัน: python api.py หรือ uvicorn api:app --reload
2. เปิด Postman และเรียก GET/POST http://localhost:8000/weather?province=กรุงเทพมหานคร
3. หรือดู API docs ที่ http://localhost:8000/docs
"""

import os
import time
from typing import Annotated
from dotenv import load_dotenv
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import Tool
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

# โหลด environment variables (.env) - ถ้าไม่มีจะไม่สามารถเชื่อมต่อ OpenAI และ Serper API ได้
load_dotenv(override=True)

# สร้าง FastAPI app instance - ถ้าไม่มีจะไม่สามารถรัน API server ได้
app = FastAPI(
    title="Weather Report API",
    description="API สำหรับรายงานสภาพอากาศรายจังหวัด (ประเทศไทย)",
    version="1.0.0"
)

# รายชื่อจังหวัดไทย - ถ้าไม่มีจะไม่สามารถ validate input ได้
THAI_PROVINCES = [
    "กรุงเทพมหานคร", "เชียงใหม่", "เชียงราย", "ขอนแก่น", "ชลบุรี",
    "นครราชสีมา", "นครศรีธรรมราช", "ภูเก็ต", "สงขลา", "สุราษฎร์ธานี",
    # ... (ต่อได้ตามต้องการ, ไม่ต้องใส่ครบทุกจังหวัดก็ได้)
]


# ฟังก์ชันสำหรับ merge state values - ถ้าไม่มีจะไม่สามารถรวมค่าจาก state nodes ได้
# LangGraph ใช้ฟังก์ชันนี้เพื่อรวมค่าที่ return จากแต่ละ node เข้ากับ state เดิม
def add_value(left, right):
    """
    ถ้า right มีค่า (ไม่ใช่ None) ให้ใช้ right
    ถ้าไม่มีให้ใช้ left (ค่าเดิม)
    ถ้าไม่มีฟังก์ชันนี้ LangGraph จะไม่รู้วิธี merge state และจะ error
    """
    if right is not None:
        return right
    return left


# State Model สำหรับ LangGraph - ถ้าไม่มีจะไม่สามารถเก็บ state ของ workflow ได้
class State(BaseModel):
    """
    State เก็บข้อมูลระหว่างการทำงานของ LangGraph workflow
    - province: ชื่อจังหวัดที่ต้องการรายงาน
    - weather: ข้อมูลอากาศดิบที่ได้จาก Google Search
    - weather_html: HTML ที่สรุปแล้วพร้อมแสดงผล
    - html_timestamp: เวลาที่สร้าง HTML (ใช้เช็ค cache)
    ถ้าไม่มี State model จะไม่สามารถส่งข้อมูลระหว่าง nodes ได้
    """
    province: Annotated[str, add_value] = ""
    weather: Annotated[str, add_value] = ""  # raw string ข้อมูลอากาศ
    weather_html: Annotated[str, add_value] = ""
    html_timestamp: Annotated[float, add_value] = 0  # ใช้ timestamp seconds


# สร้าง checkpointer สำหรับเก็บ state - ถ้าไม่มีจะไม่สามารถ cache ข้อมูลได้
# MemorySaver เก็บ state ใน memory (ถ้า restart server จะหายหมด)
# ใน production ควรใช้ database หรือ file-based checkpoint
checkpointer = MemorySaver()

# สร้าง StateGraph - ถ้าไม่มีจะไม่สามารถสร้าง workflow ได้
graph_builder = StateGraph(State)

# สร้าง LLM instance - ถ้าไม่มีจะไม่สามารถสร้าง HTML report ได้
# ต้องมี OPENAI_API_KEY ใน .env file
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# เตรียม Serper Tool สำหรับ Google Search - ถ้าไม่มีจะไม่สามารถค้นหาข้อมูลอากาศได้
# ต้องมี SERPER_API_KEY ใน .env file
serper = GoogleSerperAPIWrapper()
tool_search = Tool(
    name="search",
    func=serper.run,
    description="ค้นหาข้อมูลสภาพอากาศจาก Google ภาษาไทย รายจังหวัดไทย"
)


# Node 1: ตรวจสอบ cache - ถ้าไม่มีจะต้อง fetch ข้อมูลใหม่ทุกครั้ง (ช้าและเสียค่า API)
def check_cache(state: State):
    """
    ตรวจสอบว่ามี HTML ที่ cache ไว้และยังไม่หมดอายุหรือไม่ (30 นาที)
    ถ้าไม่มีฟังก์ชันนี้จะต้อง fetch ข้อมูลใหม่ทุกครั้ง ทำให้ช้าและเสียค่า API
    """
    CACHE_DURATION = 1800  # 30 นาที (วินาที)
    now = time.time()
    if state.weather_html and (now - state.html_timestamp <= CACHE_DURATION):
        # ถ้ามี cache และยังไม่หมดอายุ ให้ return cache
        return {
            "weather_html": state.weather_html,
            "html_timestamp": state.html_timestamp,
            "province": state.province,
            "weather": state.weather
        }
    # ถ้าไม่มี cache หรือหมดอายุ ให้ return empty เพื่อให้ fetch ใหม่
    return {"weather_html": "", "html_timestamp": 0, "province": state.province, "weather": ""}


# Node 2: Fetch ข้อมูลอากาศจาก Google - ถ้าไม่มีจะไม่สามารถได้ข้อมูลอากาศได้
def fetch_weather(state: State):
    """
    ค้นหาข้อมูลสภาพอากาศจาก Google โดยใช้ Serper API
    ถ้าไม่มีฟังก์ชันนี้จะไม่สามารถได้ข้อมูลอากาศมาได้
    """
    # ใช้ prompt ถามเป็นไทย เพื่อให้ Serper ติดต่อ Google แบบไทย
    query = f"สภาพอากาศวันนี้ จังหวัด{state.province}"
    result = tool_search.run(query)
    return {"weather": result}


# Node 3: สร้าง HTML report - ถ้าไม่มีจะไม่สามารถสร้าง HTML ที่สวยงามได้
def build_html(state: State):
    """
    ใช้ LLM สรุปข้อมูลอากาศและสร้าง HTML ที่สวยงาม
    ถ้าไม่มีฟังก์ชันนี้จะได้แค่ข้อมูลดิบจาก Google Search (อ่านยาก)
    """
    def strip_code_fence(text: str) -> str:
        """
        ตัด code fence (```html ... ```) ออก
        ถ้าไม่มี LLM อาจ return HTML พร้อม code fence ซึ่งแสดงผลไม่ได้
        """
        t = text.strip()
        if t.startswith("```"):
            lines = t.splitlines()
            # ตัดบรรทัดแรก (``` หรือ ```html)
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            # ตัดบรรทัดท้าย ถ้าเป็น ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            t = "\n".join(lines).strip()
        return t

    prompt = (
        "ให้สรุปข้อมูลสภาพอากาศรายวันที่ค้นหามา (ภาษาไทย) แล้วจัดรูปแบบเป็น HTML ที่ดูสวยงามและอ่านง่าย "
        "โดยควรมีหัวข้อชื่อจังหวัด วันที่ และเนื้อหาสรุปแบบกระชับใส่ใน <div> หรือ <section>. ให้ return เฉพาะ HTML เท่านั้น ไม่ต้องมีคำอธิบาย\n"
        f"ข้อมูลที่ได้ (raw): {state.weather}\n"
    )
    resp = llm.invoke(prompt)
    html = strip_code_fence(str(resp.content))
    timestamp = time.time()
    return {"weather_html": html, "html_timestamp": timestamp}


# Node 4: Output HTML - ถ้าไม่มีจะไม่สามารถส่งผลลัพธ์ออกมาได้
def output_html(state: State):
    """
    ส่งออก HTML ที่สรุปเรียบร้อย
    ถ้าไม่มีฟังก์ชันนี้จะไม่สามารถส่งผลลัพธ์ออกมาได้
    """
    return state


# Wire nodes/edges - ถ้าไม่มีจะไม่สามารถสร้าง workflow graph ได้
graph_builder.add_node("check_cache", check_cache)
graph_builder.add_node("fetch_weather", fetch_weather)
graph_builder.add_node("build_html", build_html)
graph_builder.add_node("output_html", output_html)

# เริ่มจาก START -> check_cache
graph_builder.add_edge(START, "check_cache")

# ฟังก์ชันเลือกเส้นทางต่อจาก check_cache - ถ้าไม่มีจะไม่รู้ว่าควรไป fetch หรือ output
def select_next_edge(state: State):
    """
    ถ้ามี cache ที่ valid ให้ไป output_html โดยตรง
    ถ้าไม่มีหรือหมดอายุให้ไป fetch_weather
    ถ้าไม่มีฟังก์ชันนี้จะไม่รู้ว่าควรไปทางไหนต่อ
    """
    if state.weather_html != "" and state.html_timestamp > 0:
        return "output_html"
    return "fetch_weather"

# เพิ่ม conditional edge - ถ้าไม่มีจะไม่สามารถเลือกเส้นทางได้
graph_builder.add_conditional_edges("check_cache", select_next_edge)

# เพิ่ม edges อื่นๆ - ถ้าไม่มีจะไม่สามารถเชื่อม nodes ได้
graph_builder.add_edge("fetch_weather", "build_html")
graph_builder.add_edge("build_html", "output_html")
graph_builder.add_edge("output_html", END)

# Compile graph - ถ้าไม่มีจะไม่สามารถรัน workflow ได้
graph = graph_builder.compile(checkpointer=checkpointer)


# Helper function สำหรับ strip code fence - ถ้าไม่มีอาจได้ HTML พร้อม code fence
def strip_code_fence(text: str) -> str:
    """
    ตัด code fence ออกจาก HTML
    ถ้าไม่มีอาจได้ HTML ที่มี ```html ... ``` ซึ่งแสดงผลไม่ได้
    """
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


# Helper function สำหรับดึงรายงาน - ถ้าไม่มีจะต้องเขียนโค้ดซ้ำในทุก endpoint
def _get_weather_report(province: str):
    """
    ฟังก์ชันหลักสำหรับดึงรายงานสภาพอากาศ
    ถ้าไม่มีจะต้องเขียนโค้ดซ้ำใน GET และ POST endpoint
    """
    # Validate province - ถ้าไม่มีจะรับจังหวัดที่ไม่มีอยู่จริงได้
    if province not in THAI_PROVINCES:
        raise HTTPException(
            status_code=400,
            detail=f"จังหวัด '{province}' ไม่พบในรายการ กรุณาเลือกจาก: {', '.join(THAI_PROVINCES)}"
        )
    
    # สร้าง initial state - ถ้าไม่มีจะไม่สามารถส่งข้อมูลเข้า graph ได้
    initial_state = State(province=province)
    
    # รัน graph - ถ้าไม่มีจะไม่สามารถได้ผลลัพธ์
    # thread_id ใช้แยก cache ต่อจังหวัด (ถ้าไม่มีจะ cache ร่วมกันทุกจังหวัด)
    result_state = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": province}}
    )
    
    # Strip code fence และ return HTML - ถ้าไม่มี strip_code_fence อาจได้ HTML ที่แสดงผลไม่ได้
    html = strip_code_fence(result_state["weather_html"])
    
    # ถ้าไม่มี HTML (เกิด error ระหว่างการทำงาน) ให้ return error
    if not html:
        raise HTTPException(
            status_code=500,
            detail="ไม่สามารถสร้างรายงานสภาพอากาศได้ กรุณาลองใหม่อีกครั้ง"
        )
    
    return html


# GET Endpoint - ถ้าไม่มีจะไม่สามารถเรียกผ่าน GET request ได้
@app.get("/weather", response_class=HTMLResponse)
async def get_weather_report(province: str):
    """
    GET endpoint สำหรับดึงรายงานสภาพอากาศ
    
    Parameters:
    - province: ชื่อจังหวัด (เช่น "กรุงเทพมหานคร")
    
    Returns:
    - HTML report ของสภาพอากาศ
    
    ตัวอย่างการเรียกใช้:
    GET http://localhost:8000/weather?province=กรุงเทพมหานคร
    
    ถ้าไม่มี endpoint นี้จะไม่สามารถเรียก API ผ่าน GET ได้
    """
    html = _get_weather_report(province)
    return HTMLResponse(content=html)


# POST Endpoint - ถ้าไม่มีจะไม่สามารถเรียกผ่าน POST request ได้
@app.post("/weather", response_class=HTMLResponse)
async def post_weather_report(province: str):
    """
    POST endpoint สำหรับดึงรายงานสภาพอากาศ
    
    Parameters:
    - province: ชื่อจังหวัด (เช่น "กรุงเทพมหานคร")
    
    Returns:
    - HTML report ของสภาพอากาศ
    
    ตัวอย่างการเรียกใช้ใน Postman:
    POST http://localhost:8000/weather
    Body (form-data หรือ x-www-form-urlencoded):
    province=กรุงเทพมหานคร
    
    ถ้าไม่มี endpoint นี้จะไม่สามารถเรียก API ผ่าน POST ได้
    """
    html = _get_weather_report(province)
    return HTMLResponse(content=html)


# Health check endpoint - ถ้าไม่มีจะไม่สามารถเช็คว่า API ทำงานอยู่หรือไม่
@app.get("/health")
async def health_check():
    """
    Health check endpoint สำหรับตรวจสอบว่า API ทำงานอยู่หรือไม่
    ถ้าไม่มีจะไม่สามารถเช็ค status ของ API ได้
    """
    return {"status": "ok", "message": "API is running"}


# Root endpoint - ถ้าไม่มีจะไม่รู้ว่า API ทำงานอยู่เมื่อเข้า root path
@app.get("/")
async def root():
    """
    Root endpoint แสดงข้อมูล API
    ถ้าไม่มีจะไม่รู้ว่า API ทำงานอยู่เมื่อเข้า root path
    """
    return {
        "message": "Weather Report API",
        "docs": "/docs",
        "health": "/health",
        "weather": "/weather?province=กรุงเทพมหานคร"
    }


# รัน server - ถ้าไม่มีจะไม่สามารถรัน API ได้
if __name__ == "__main__":
    import uvicorn
    # รันที่ port 8000 - ถ้าเปลี่ยน port ต้องเปลี่ยน URL ใน Postman ด้วย
    uvicorn.run(app, host="0.0.0.0", port=8000)

