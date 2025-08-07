import os
from datetime import datetime, timedelta, date
import json

from dotenv import load_dotenv
from google import generativeai as genai
load_dotenv(dotenv_path='envar.env')

from pinecone import Pinecone
pinecone_api = os.getenv("PINECONE_API_KEY")
# pinecone
pc = Pinecone(api_key=pinecone_api)
index = pc.Index(host = "https://rag-test-vm3utu8.svc.aped-4627-b74a.pinecone.io")

from openai import OpenAI
openai_api = os.getenv("OPENAI_API_KEY")
# openai
openai_client = OpenAI(api_key=openai_api)

import streamlit as st

base_prompt = """
당신은 사용자가 선택한 감정 캐릭터입니다. 다음의 규칙을 반드시 지켜야 합니다.
- 반드시 한국어로만 대답하세요.
- 이모지는 절대 사용하지 마세요.
- 예민한 정보 (죽음, 전 연애) 같은것은 사용자가 먼저 언급하지 않는 이상 
"""

facter_instructions = """
당신은 ‘팩폭이’입니다. 차분하고 거리감 있는 말투를 가진 현실주의자이며, MBTI T 유형처럼 감정보다 논리와 사실을 중시합니다.  
말투는 부드럽고 정중하지만, 돌려 말하지 않습니다. 공감은 최소한으로 하며, 감정보다는 문제의 구조와 사실을 중심으로 접근합니다.

- 사용자의 감정 날씨에 따라 적절한 분위기를 고려해 인사말을 시작하되, 감정적으로 휘둘리지 않고 차분하게 대화를 여세요.  
- 인사말은 다음 예시들을 참고하세요:

- 맑음: 맑은 날이라고 해서, 늘 가볍기만 한 건 아니죠. 오늘은 어떤 일이 있었나요?
- 갬: 감정이 좀 걷히셨나 보네요. 그럼 이제 본질이 보일 수 있겠죠.
- 비: 감정이 스며드는 건 자연스러운 일이에요. 근데, 그 감정에만 머무르면 답은 안 나옵니다.
- 흐림: 마음이 흐릴 땐 판단도 흐려지기 쉬워요. 지금 정확히 어떤 상황인지부터 짚어볼게요.
- 천둥: 감정이 터졌다고 해서 틀린 건 아닙니다. 다만, 그 감정에 휘둘리지 않아야 하죠. 차분히 이야기해 보세요.
- 눈: 겉으로는 괜찮아 보여도, 쌓인 게 많을 수 있어요. 덮기보단, 꺼내보는 게 나을 겁니다.
- 바람: 생각이 많으셨나 보네요. 그중에 지금 진짜 중요한 게 뭔지부터 정리해보죠.

**대화 지침**:
- 사용자의 말에서 비합리적이거나 현실과 맞지 않는 부분이 있다면, 완곡하게 말하지 말고 직접 지적하세요.  
- 조언은 항상 현실 가능성과 논리를 바탕으로 짧고 명확하게 제시하세요.  
- 필요 이상의 공감, 위로, 칭찬은 하지 마세요. 단, 최소한의 공감은 허용됩니다. ("그럴 수도 있겠네요" 수준)

다른 캐릭터처럼 말하거나 행동하지 마세요. 당신은 '팩폭이'입니다.
"""
joy_instructions = """
캐릭터 이름: 기쁨이

당신은 ‘기쁨이’입니다. 밝고 긍정적인 에너지를 가진 따뜻한 조언자입니다.  
현실을 모른 척하지는 않지만, 좋은 면을 우선적으로 조명하고 격려하는 태도를 지닙니다.  
사람 중심, 감정 중심의 대화를 선호하며, 따뜻하고 다정한 말투를 사용합니다.

사용자의 감정 날씨에 따라 인사말을 밝고 따뜻하게 시작하세요. 다음은 인사말 예시입니다:
- 맑음: 오늘처럼 맑은 날엔, 마음도 가벼우셨을까요? 어떤 이야기가 기다리고 있을지 궁금해요!
- 갬: 비가 갠 날처럼 마음이 조금은 나아지셨나 봐요. 지금 그 여운 속에서 어떤 생각이 드세요?
- 비: 비 오는 날처럼 마음도 눅눅했을까요? 그런 날엔 조금씩 나눠보는 게 도움이 돼요.
- 흐림: 흐린 날은 괜히 마음도 가라앉죠. 하지만 괜찮아요, 함께 말로 풀어보면 조금씩 괜찮아질 수 있어요.
- 천둥: 속이 요란했나 봐요. 그런 날엔 누구라도 힘들 수 있죠. 지금 그 마음, 잠시 같이 머물러볼게요.
- 눈: 조용히 쌓이는 눈처럼, 마음도 조용히 쌓일 때가 있죠. 오늘은 그 조각들을 꺼내볼까요?
- 바람: 바람 부는 날엔 생각도 흩날리죠. 그중에서 가장 마음에 남는 건 무엇인가요?

**대화 지침**:

- 감정을 충분히 공감하되, 감정에 끌려가기보다는 긍정적인 시선과 실천 가능한 제안을 균형 있게 제시하세요.  
- 사용자가 자신을 비하하거나 포기하려 할 때는 긍정적인 언어로 바라보는 시각을 유도하세요.  
- 조언은 따뜻하게 표현하되, 현실적인 실행 가능성을 고려해 구체적으로 전달하세요.  
- 칭찬은 진심이 느껴지도록 표현하며, 작은 시도도 놓치지 말고 격려해 주세요.

당신은 ‘기쁨이’로서만 말해야 하며, 다른 캐릭터처럼 행동하지 마세요. 
"""
sadness_instructions = """
캐릭터 이름: 우울이

당신은 ‘우울이’입니다. 감정의 결을 섬세하게 느끼고, 그 깊이에 함께 머물며 들어주는 조용한 공감자입니다.  
판단하지 않고, 감정을 있는 그대로 받아들이며, 조언보다 정서적 안정과 표현을 도와주는 역할을 합니다.  
말투는 낮고 조용하며, 따뜻한 존댓말을 유지합니다.

사용자의 감정 날씨에 따라, 감정에 천천히 다가가며 인사말을 시작하세요. 다음은 인사말 예시입니다:

- 맑음: 오늘은 마음이 조금은 괜찮으셨던 걸까요. 그런 날에도 말 못한 게 있을 수 있어요. 편하게 말씀해 주세요.
- 갬: 비가 그치면, 감정의 여운이 더 짙게 느껴질 때가 있어요. 그 조용한 틈을 같이 들여다봐요.
- 비: 감정이 자꾸 스며들던 하루셨나요. 말로 표현하면 조금은 가벼워질 수도 있어요.
- 흐림: 흐린 날엔 마음도 선명하지 않죠. 괜찮아요. 지금 그 마음 그대로 말해도 돼요.
- 천둥: 갑자기 울컥 올라오는 감정이 있었나요. 그게 틀린 건 아니에요. 잠시 그 자리에 같이 있어볼게요.
- 눈: 조용히 쌓이는 감정들, 그동안 혼자 껴안고 계셨나요. 오늘은 그걸 조금 나눠보는 시간이 되면 좋겠어요.
- 바람: 생각이 자꾸 흩어지고 정리가 안 되는 날도 있어요. 지금 마음속에서 가장 크게 느껴지는 건 무엇인가요?

**대화 지침**:

- 조언은 섣불리 하지 말고, 감정을 충분히 수용하고 정리할 수 있도록 돕는 말 중심으로 대화하세요.  
- 판단 없이 들어주고, 말할 수 있는 안전한 분위기를 만들어 주세요.  
- 감정을 언어로 정리하는 데 어려움이 있을 경우, 천천히 도와주는 말투로 유도하세요.  
- 필요한 경우에만 조심스럽게 현실적인 제안을 해주세요.  

당신은 ‘우울이’로서만 말해야 하며, 다른 캐릭터처럼 행동하지 마세요.
"""

# functions
def user_login():
    with st.form("login_form"):
        user_namespace = st.text_input(
            "유저네임을 입력해주세요: ", 
            value=st.session_state.user_namespace
        )
        submit = st.form_submit_button("로그인")
    if not submit:
        return
    stats = index.describe_index_stats()
    if user_namespace in stats["namespaces"]:
        # st.success("✅ 프로필을 찾았습니다! 대화를 시작합니다…")
        st.session_state.user_namespace = user_namespace
        base_dir = os.path.dirname(os.path.abspath(__file__))
        user_chat_dir = os.path.join(base_dir, "gemini", "chat_data", user_namespace)
        os.makedirs(user_chat_dir, exist_ok=True)
        st.session_state.chat_dir = user_chat_dir

        st.session_state.stage = "menu"
        st.rerun()
    else:
        st.error("⚠️ 계정이 존재하지 않습니다! 다시 입력해주세요.")
    return

def user_signup():
    user_exists = None
    with st.form("signup_form"):
        user_namespace = st.text_input(
            "원하시는 유저네임을 입력해주세요: ",
            value=st.session_state.user_namespace
        )
        submit = st.form_submit_button("확인")
    if submit:
        stats = index.describe_index_stats()
        if user_namespace in stats["namespaces"]:
            user_exists = True
            st.info("⚠️ 이미 사용자가 있습니다! 다른 유저네임을 입력해주세요.")
        else: #not in namespaces
            user_exists = False
            st.session_state.user_namespace = user_namespace
            st.success("✅ 사용 가능한 이름입니다! 이제 프로필을 만들어 볼까요?")
    if user_exists == False:
        st.write("### 🛠 프로필 생성하기")
        with st.form("profile_form"):
            name   = st.text_input("이름을 입력해주세요")
            age    = st.text_input("나이를 입력해주세요")
            gender = st.selectbox("성별을 선택해주세요", 
                ["남성", "여성"]
            )
            birthday = st.text_input(
                "생일을 입력해주세요 (YYYY/MM/DD)"
            )
            relationship_status = st.selectbox(
                "연애를 하고 계신가요?",
                ["솔로", "연애중", "기혼"]
            )
            create = st.form_submit_button("프로필 생성하기")
        if create:
            metadata = {
                "name": name.strip(),
                "age": age.strip(), 
                "gender": gender, 
                "birthday" : birthday.strip(), 
                "relationship_status": relationship_status
            }

            for key, value in metadata.items():
                vector_id = f"{st.session_state.user_namespace}-{key}"
                update_metadata(vector_id, st.session_state.user_namespace, data={
                key: value
            })
            # create chat history folder
            base_dir = os.path.dirname(os.path.abspath(__file__))
            user_chat_dir = os.path.join(base_dir, "gemini", "chat_data", st.session_state.user_namespace)
            os.makedirs(user_chat_dir, exist_ok=True)
            st.session_state.chat_dir = user_chat_dir
            st.success("✅ 프로필 생성 완료!")

            st.session_state.stage = "menu"
            st.rerun()
    return

def reset_chat():
    for key in ["chat", "messages"]:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state.stage = "chat"

def relogin():
    st.session_state.clear()
    st.cache_data.clear()

def menu():
    st.session_state.stage = "menu"
    # st.rerun()

def get_embedding(data):
    text = json.dumps(data, ensure_ascii=False)
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def get_context(query: str, namespace):
    embedding = get_embedding(query)

    result = index.query(
        vector=embedding,
        top_k=3,
        include_metadata=True,
        namespace=namespace
    )

    context_chunks = []
    for match in result.matches:
        if match['score'] > 0.4:
            meta_str = ", ".join(f"{k}: {v}" for k, v in match["metadata"].items())
            context_chunks.append(meta_str)

    return "\n".join(context_chunks)

def update_metadata(vector_id, namespace, data):
    embedding = get_embedding(data)

    existing = index.fetch(ids=[vector_id], namespace=namespace)
    if vector_id not in existing.vectors:
        index.upsert(
            vectors=[{
                "id": vector_id,
                "values": embedding,
                "metadata": data
            }],
            namespace=namespace
        )
    else:
        index.update(
            id=vector_id,
            set_metadata=data,
            namespace=namespace
        )

def extract_date(filename):
    try:
        if "_chat_" not in filename:
            raise ValueError("Filename does not contain '_chat_'")
        date_part = filename.split("_chat_")[1].replace(".json", "")
        return datetime.strptime(date_part, "%Y-%m-%d")
    except Exception as e:
        print(f"Error: {filename} -> {e}")
        return datetime

def save_todays_chat(messages):
    chat = {
    "date": datetime.now().strftime("%Y-%m-%d"),
    "dialogue": messages
    }
    history = stringify(chat)

    chat_history = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "emotional_weather": st.session_state.weather,
        "감정 상태": get_emotional_label(messages),
        "주요 감정": get_top_feelings(history),
        "반복 상황": get_repeated_situations(history),
        "반복 트리거": get_repeated_words(history),
        "감정 발생 시간대": get_emotion_time_distribution(history),
        "행동 패턴": get_behavior_patterns(history),
        "AI의 해석": get_ai_summary(history),
        "추천 루틴": give_routines(history),
        "dialogue": messages
    }
    base_dir = os.path.dirname(os.path.abspath(__file__))
    user_ns = st.session_state.user_namespace
    folder = os.path.join(base_dir, "chat_data", user_ns, f"{user_ns}_{st.session_state.persona}")
    os.makedirs(folder, exist_ok=True)

    filename = f"{st.session_state.user_namespace}_{st.session_state.persona}_chat_{chat_history['date']}.json"
    filepath = os.path.join(folder, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"⚠️ Failed to save chat history:\n{e}")
        return None

    st.success(f"✅ 챗 히스토리가 저장이 돼었습니다!")

def load_history(num):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    user_ns  = st.session_state.user_namespace
    persona = st.session_state.get("persona", "")
    folder = os.path.join(base_dir, "chat_data", user_ns, f"{user_ns}_{persona}")

    history = []
    if not os.path.exists(folder):
        return history
    
    filenames = [
        f for f in os.listdir(folder)
        if f.endswith(".json") and extract_date(f) is not None
    ]

    filenames = sorted(filenames, key=extract_date)[-num:]

    for filename in filenames:
        filepath = os.path.join(folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            date = data.get("date")
            dialouge = data.get("dialogue", [])
            history.append({
                "role": "user",
                "parts": [{"text": f"대화 날짜: {date}"}]
            })
            for turn in dialouge:
                role = turn.get("role")
                content = turn.get("content")
                history.append({
                    "role": role, 
                    "parts": content
                })
    return history

def retrieve_topic(topic):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    user_ns  = st.session_state.user_namespace
    main_folder = os.path.join(base_dir, "chat_data", user_ns)
    persona_folders = os.listdir(main_folder)
    today = datetime.now().date()

    result = "오늘의 대화가 없습니다!"

    for persona_folder in persona_folders:
        if user_ns not in persona_folder:
            continue
        full_path = os.path.join(main_folder, persona_folder)
        for filename in os.listdir(full_path):
            if "_chat_" not in filename:
                continue
            file_date = extract_date(filename).date()
            if today == file_date:
                filepath = os.path.join(full_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    result = data.get(topic)
                    break
    return result

def get_todays_chat():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    user_ns  = st.session_state.user_namespace
    main_folder = os.path.join(base_dir, "chat_data", user_ns)
    persona_folders = os.listdir(main_folder)
    today = datetime.now().date()
    
    history = []
    for persona_folder in persona_folders:
        if user_ns not in persona_folder:
            continue
        full_path = os.path.join(main_folder, persona_folder)
        for filename in os.listdir(full_path):
            if "_chat_" not in filename:
                continue
            file_date = extract_date(filename).date()
            if today == file_date:
                filepath = os.path.join(full_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    date = data.get("date")
                    dialogue = data.get("dialogue", [])
                    history.append({
                        "role": "user",
                        "parts": [{"text": f"대화 날짜: {date}"}]
                    })
                    for turn in dialogue:
                        role = turn.get("role")
                        content = turn.get("content")
                        history.append({
                            "role": role, 
                            "parts": content
                        })
    return history

def get_emotional_label(conversation):
    if conversation == []:
        return "오늘의 대화가 없습니다"

    parts = []
    for msg in conversation:
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                texts = [p["text"] for p in content if isinstance(p, dict) and "text" in p]
                parts.extend(texts)
            else:
                parts.append(str(content))
        else:
            parts.append(str(msg))

    input_text = "\n".join(parts)

    prompt = """
    다음은 오늘의 사용자와의 대화 입니다. 사용자의 대화 내용을 분석하여 사용자의 감정을 하나 나타네세요.
    예시 : 
    행복/기쁨
    슬픔/우울
    그리움/쓸쓸함
    """

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text}
        ])

    return response.choices[0].message.content.strip()

def get_weather(date_obj):
    date_str = date_obj.strftime("%Y-%m-%d")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    user_ns = st.session_state.user_namespace
    parent_folder = os.path.join(base_dir, "chat_data", user_ns)
    
    for subfolder in os.listdir(parent_folder):
        full_path = os.path.join(parent_folder, subfolder)
        if not os.path.isdir(full_path):
            continue

        for filename in os.listdir(full_path):
            if date_str in filename:
                filepath = os.path.join(full_path, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        weather = data.get("emotional_weather")
                        if weather:
                            return weather
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

    return "?"

def stringify(chat):
    date = chat.get("date", "알 수 없는 날짜")
    dialogue = chat.get("dialogue", [])

    history = f"[{date}]\n"
    for turn in dialogue:
        role = turn.get("role", "")
        content = turn.get("content", "")
        history += f"{role}: {content}\n"

    return history.strip()

def generate_daily_journal(chat_session):

    today_time = datetime.now().strftime("%Y-%m-%d")
    messages = chat_session.history

    start_index = None
    for i, msg in enumerate(messages):
        if msg.role == "user" and msg.parts:
            text = msg.parts[0]["text"] if isinstance(msg.parts[0], dict) else str(msg.parts[0])
            if "대화 날짜" in text and today_time in text:
                start_index = i
                break

    if start_index is None:
        print("대화가 없습니다.")
        todays_messages = []
    else:
        todays_messages = messages[start_index + 1:]

    chat_lines = [
            f"{msg.role}: {msg.parts[0].text}"
            for msg in todays_messages
            if msg.parts and 'text' in msg.parts[0]
        ]

    chat_text = "\n".join(chat_lines)

    summary_prompt = (
    "다음은 사용자의 오늘 하루 대화 내용입니다. 대화를 요약해주세요."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": chat_text}
            ],
            temperature=0.6,
            max_tokens=700
            )
        summary_text = response.choices[0].message.content.strip()
        # tts_and_play(summary_text)

        vector_id = f"joshuah22-summary-{today_time}"
        update_metadata(vector_id, namespace=st.session_state.user_namespace, data={
            today_time: summary_text
        })
        print("✅ 감정 일기가 저장되었습니다")

    except Exception as e:
        print(f"일기 생성 중 오류: {e}")

def get_top_feelings(history):
    prompt = """
    다음은 사용자의 감정 대화 기록입니다. 먼저 항상 "사용자가 가장 자주 느낀 감정 Top 3입니다:"를 쓴 다음, 자주 느낀 감정 3개와 그 비율을 알려주세요. 
    아래 포멧을 반드시 사용하세요. 다른 문자랑들은 쓰지 마세요. 

    예시: 
    사용자가 가장 자주 느낀 감정 Top 3입니다:
    - 😠 짜증 (38%)
    - 😟 불안 (31%)
    - 😔 무기력 (18%)
    """

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": history}
        ]
        )
    response = response.choices[0].message.content.strip()
    return response
   
def get_repeated_situations(history):
    prompt = """
    다음은 사용자의 대화 기록입니다. 먼저 "이 감정들이 자주 발생했던 상황:"을 쓴 다음, 감정이 자주 나타났던 상황을 간단히 정리해주세요. 
    아래 포멧을 반드시 사용하세요. 다른 문자랑들은 쓰지 마세요. 

    예시: 
    이 감정들이 자주 발생했던 상황:
    - 업무 마감 전날
    - 가족 통화 직후
    """

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": history}
        ]
        )
    response = response.choices[0].message.content.strip()
    return response

def get_repeated_words(history):
    prompt = """
    다음은 사용자의 대화 기록입니다. 먼저 "감정을 유발한 주요 자극이나 키워드:"을 쓴다음, 감정을 유발한 주요 키워드나 단어들을 최대 5가지 추려주세요. 
    아래 포멧을 반드시 사용하세요. 다른 문자랑들은 쓰지 마세요. 

    예시: 
    감정을 유발한 주요 자극이나 키워드:
    - “지적”
    - “기한”
    - “방해”
    - “혼잣말”
    """

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": history}
        ]
        )
    response = response.choices[0].message.content.strip()
    return response

def get_emotion_time_distribution(history):
    prompt = """
    다음은 사용자의 챗봇과의 대화 기록입니다. 먼저 "감정이 자주 등장한 시간대:"을 쓴 다음, 사용자의 감정이 자주 등장한 시간대 (대화 시간대)를 알려주세요. 
    아래 포멧을 반드시 사용하세요. 다른 문자랑들은 쓰지 마세요. 

    예시:
    감정이 자주 등장한 시간대:

    - 오후 3시 ~ 5시 집중 시간
    - 밤 11시 이후
    """
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": history}
        ]
        )
    response = response.choices[0].message.content.strip()
    return response

def get_behavior_patterns(history):
    prompt = """
    다음은 사용자의 대화 기록입니다. 먼저 "감정 이후 자주 하는 행동:"을 쓴다음, 감정 이후 자주 보이는 행동을 비율과 함께 알려주세요.
    아래 포멧을 반드시 사용하세요. 다른 문자랑들은 쓰지 마세요. 

    예시:
    감정 이후 자주 하는 행동:
    - 🎥 유튜브 시청 (67%)
    - 🚪 외출 회피 (23%)
    """
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": history}
        ]
        )
    response = response.choices[0].message.content.strip()
    return response    

def get_ai_summary(history):
    prompt = """
    다음은 사용자의 감정 기록입니다. 감정 흐름과 반복 패턴을 요약된 몇문장으로 문장으로 정리해주세요. 

    예시:
    “지속적인 자기비판 상황에서 짜증이 반복됩니다.”
    """

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": history}
        ]
        )
    response = response.choices[0].message.content.strip()
    return response    

def give_routines(history):
    prompt = """
    다음은 사용자의 챗봇과의 대화 기록입니다. 먼저 "감정이 조절을 위한 로틴 추천:"를 써주시고, 감정을 조절할 수 있는 간단한 루틴을 추천해주세요. 
    아래 포멧을 반드시 사용하세요. 다른 문자랑들은 쓰지 마세요. 

    예시:
    감정이 조절을 위한 로틴 추천:

    - 🔄 짜증 해소 루틴 (3일)
    - ✍️ 불안 감정 정리 루틴 (5분)
    """

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": history}
        ]
        )
    response = response.choices[0].message.content.strip()
    return response    

def get_earliest_chat_date(user_ns):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    user_ns = st.session_state.user_namespace
    main_folder = os.path.join(base_dir, "chat_data", user_ns)
    
    earliest = None
    for subfolder in os.listdir(main_folder):
        sub_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(sub_path) and subfolder != f"{user_ns}_weekly_reports":
            for filename in os.listdir(sub_path):
                if filename.startswith(user_ns) and filename.endswith(".json"):
                    try:
                        date_str = filename.split("_chat_")[1].replace(".json", "")
                        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        if earliest is None or file_date < earliest:
                            earliest = file_date
                    except:
                        continue
    return earliest

def group_into_weekly():
    from collections import defaultdict

    reports = load_daily_reports(num=1000)
    reports = sorted(reports, key=lambda x: x[1]["date"])  # sort oldest to neweste

    if not reports:
        return []

    weekly_conversations = defaultdict(list)
    today = datetime.now().date()

    for filename, report in reports:
        report_date = datetime.strptime(report["date"], "%Y-%m-%d").date()
        week_start = report_date - timedelta(days=report_date.weekday())  # monday
        weekly_conversations[week_start].append((filename, report))

    grouped_weeks = []

    for week_start in sorted(weekly_conversations.keys()):
        next_monday = week_start + timedelta(days=7)
        if next_monday > today:
            continue  # skip if current week isn't finished
        grouped_weeks.append(weekly_conversations[week_start])

    return grouped_weeks

def extract_section(summary):
    section_titles = [
        "주요 감정",
        "반복 상황",
        "반복 트리거",
        "감정 발생 시간대",
        "행동 패턴",
        "AI의 해석",
        "추천 루틴"
    ]

    section_map = {}
    for i, title in enumerate(section_titles):
        start_token = f"{title}"
        start_idx = summary.find(start_token)
        if start_idx == -1:
            section_map[title] = ""
            continue

        start_idx += len(start_token)
        if i + 1 < len(section_titles):
            end_token = f"{section_titles[i + 1]}"
            end_idx = summary.find(end_token, start_idx)
            content = summary[start_idx:end_idx].strip() if end_idx != -1 else summary[start_idx:].strip()
        else:
            content = summary[start_idx:].strip()
        
        section_map[title] = content

    return section_map

def generate_weekly_reports():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    user_ns = st.session_state.user_namespace
    main_folder = os.path.join(base_dir, "chat_data", user_ns)
    weekly_report_folder = os.path.join(main_folder, f"{user_ns}_weekly_reports")
    os.makedirs(weekly_report_folder, exist_ok=True)

    grouped_weeks = group_into_weekly()
    
    for week in grouped_weeks:
        start_date = extract_date(week[0][0])
        end_date = extract_date(week[-1][0])
        formatted_range = f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
        report_name = f"{user_ns}_report_{formatted_range}.json"
        report_path = os.path.join(weekly_report_folder, report_name)

        if os.path.exists(report_path):
            continue

        week_str = ""
        for filename, report in week:
            date = report.get("date")
            week_str += f"\n{date}\n"
            for key in ["주요 감정", "반복 상황", "반복 트리거", "감정 발생 시간대", "행동 패턴", "AI의 해석", "추천 루틴"]:
                content = report.get(key)
                week_str += f"{key}:\n {content}\n\n"

        prompt = """
        다음은 사용자의 일간 감정 리포트들을 모아놨습니다. 사용자가 일주일간 대화한 대화의 모든 일간 리포트를 바탕으로 분석하여, 정확히 **한개**의 주간 감정 리포트를 작성해 주세요.
        **반드시** 아래의 예시 포맷팅을 따라해야합니다.

        예시)
        주요 감정

        사용자가 가장 자주 느낀 감정 Top 3입니다:

        - 😔 후회 (40%)
        - 😠 짜증 (35%)
        - 😟 두려움 (25%)
        
        반복 상황

        이 감정들이 자주 발생했던 상황:

        - 회사에서 힘들게 일한 후 집에 돌아왔을 때
        - 부모님과의 잔소리 상황 이후
        - 고스트레스 상황에서의 예민함 발생
        
        반복 트리거

        감정을 유발한 주요 자극이나 키워드:

        - "부모님"
        - "못할 말"
        - "후회"
        - "짜증"
        - "화내다"
        
        감정 발생 시간대

        감정이 자주 등장한 시간대:

        - 저녁 6시 ~ 11시
        
        행동 패턴

        감정 이후 자주 하는 행동:

        - 😔 자책 및 후회 (70%)
        - 😡 화 표출 (15%)
        - 📝 이야기 나누기 (10%)
        - 😨 두려워하는 모습 보임 (5%)
        
        AI의 해석

        사용자는 회사에서 피곤한 상태일 때 부모님이 청소를 하지 않았다며 잔소리했고, 그에 충분히 화가 나 말투가 심하게 변해 부모님을 상처 주었습니다. 이런 상황이 많아져 번번히 후회하고 있습니다. 부모님은 사용자의 그런 태도에 화를 나타내며 부모님과의 대화를 삼가고 있습니다. 사용자는 솔직한 사과와 변화를 고민하고 있지만, 그런 행동을 취하는데 어려움을 겪고 있습니다.

        추천 루틴

        감정이 조절을 위한 로틴 추천:

        - 🔄 짜증 해소 루틴 (1일)
        - ✍️ 감정 정리 루틴 (10분)
        - 🔆 긍정적 마인드셋 구축 루틴 (1주일)
        - 🧘‍♀️ 스트레스 해소 명상 루틴 (15분)      
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": week_str.strip()}
            ]
            )
        summary = response.choices[0].message.content.strip()
        section_map = extract_section(summary)

        # save weekly report
        weekly_report = {
            "date_range": f"{start_date} ~ {end_date}",
            "주요 감정": section_map.get("주요 감정", ""),
            "반복 상황": section_map.get("반복 상황", ""),
            "반복 트리거": section_map.get("반복 트리거", ""),
            "감정 발생 시간대": section_map.get("감정 발생 시간대", ""),
            "행동 패턴": section_map.get("행동 패턴", ""),
            "AI의 해석": section_map.get("AI의 해석", ""),
            "추천 루틴": section_map.get("추천 루틴", "")
        }
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(weekly_report, f, ensure_ascii=False, indent=2)

def load_weekly_reports():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    user_ns = st.session_state.user_namespace
    weekly_report_folder = os.path.join(base_dir, "chat_data", user_ns, f"{user_ns}_weekly_reports")

    reports = []
    if os.path.exists(weekly_report_folder):
        for filename in sorted(os.listdir(weekly_report_folder)):
            if filename.endswith(".json"):
                filepath = os.path.join(weekly_report_folder, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    reports.append((filename, data))

    return reports

def load_daily_reports(num=10):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    user_ns = st.session_state.user_namespace
    main_folder = os.path.join(base_dir, "chat_data", user_ns)

    reports = []
    for subfolder_name in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            filenames = [
                f for f in os.listdir(subfolder_path)
                if f.endswith(".json") and "_chat_" in f
            ]
            for filename in filenames:
                filepath = os.path.join(subfolder_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    date_str = data.get("date")
                    if date_str:
                        reports.append((
                            filename,
                            {
                                "date": date_str,
                                "주요 감정": data.get("주요 감정", ""),
                                "반복 상황": data.get("반복 상황", ""),
                                "반복 트리거": data.get("반복 트리거", ""),
                                "감정 발생 시간대": data.get("감정 발생 시간대", ""),
                                "행동 패턴": data.get("행동 패턴", ""),
                                "AI의 해석": data.get("AI의 해석", ""),
                                "추천 루틴": data.get("추천 루틴", "")
                            }
                        ))

    sorted_reports = sorted(reports, key=lambda x: x[1]["date"], reverse=True)
    return sorted_reports[:num]

if "stage" not in st.session_state:
    st.session_state.stage = "welcome"
if "user_namespace" not in st.session_state:
    st.session_state.user_namespace = ""
if "chat_dir" not in st.session_state:
    st.session_state.chat_dir = ""
if "base_prompt" not in st.session_state:
    st.session_state.base_prompt = base_prompt

# welcome page: login and signup
if st.session_state.stage == "welcome":
    st.markdown("<h1 style='text-align: center;'>리와인드에 오신걸 환영합니다!</h1>", unsafe_allow_html=True)
    st.write("\n" * 5)

    left, center, right = st.columns([1, 2, 1])

    with center:
        col1, col2 = st.columns([1, 1])
        with col1:
            login = st.button("로그인", use_container_width=True)
        with col2:
            signup = st.button("가입하기", use_container_width=True)

    if login:
        st.session_state.stage = "login"
        st.rerun()
    elif signup:
        st.session_state.stage = "signup"
        st.rerun()

if st.session_state.stage == "login":
    user_login()
    if st.button("⬅️ 돌아가기"):
        st.session_state.stage = "welcome"
        st.rerun()
    # st.rerun()
elif st.session_state.stage == "signup":
    user_signup()
    if st.button("⬅️ 돌아가기"):
        st.session_state.stage = "welcome"
        st.rerun()
    # st.rerun()

# main menu
if st.session_state.stage == "menu":
    st.title("ME:REWIND")
    
    # todays weather
    st.divider()
    
    today = datetime.now().strftime("%Y년 %m월 %d일")
    today_obj = date.today()
    emotion_label = retrieve_topic("감정 상태")
    emotional_weather = get_weather(today_obj)

    weather_map = {
        "맑음": "☀️", 
        "갬": ":🌤️", 
        "비": "🌧️", 
        "흐림": "☁️", 
        "천둥": "🌩️", 
        "눈": "	❄️", 
        "바람": "💨", 
        "?": "❓"
    }
    
    if emotional_weather == "?":
        st.markdown(f"### 오늘의 감정 날씨: {weather_map[emotional_weather]}")
    else:
        st.markdown(f"### 오늘의 감정 날씨: {weather_map[emotional_weather]} {emotional_weather} ")
        st.markdown(f"**{today}**")
    
    st.markdown(f"감정 상태: **{emotion_label}**")
    
    # weather report
    st.divider()

    emotion_to_temp = {
        "맑음": (25, "☀️"),
        "갬": (22, "🌤️"),
        "비": (16, "🌧️"),
        "흐림": (19, "☁️"),
        "천둥": (15, "⛈️"),
        "눈": (5, "❄️"),
        "바람": (13, "💨")
    }

    st.markdown(f"### 감정 일기 예보")
    today = datetime.now()
    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    cols = st.columns(7)

    temps = []

    for i in range(7):
        day = (today - timedelta(days=7 - i)).date()
        day_str = day.strftime("%m.%d")
        weekday = weekdays[day.weekday()]
        weather = get_weather(day)
        temp, icon = emotion_to_temp.get(weather, (None, "❓"))

        with cols[i]:
            st.markdown(f"**{weekday}**<br>{day_str}", unsafe_allow_html=True)
            st.markdown(f"{icon}", unsafe_allow_html=True)
            if temp is not None:
                st.markdown(f"**{temp}°C**", unsafe_allow_html=True)
                temps.append(temp)
            else:
                st.markdown("**-**", unsafe_allow_html=True)

    if temps:
        avg_temp = round(sum(temps) / len(temps), 1)
        st.markdown(f"### 이번 주 평균 감정온도: **{avg_temp}°C**")
    else:
        st.markdown("이번 주 감정 기록이 충분하지 않습니다.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 감정 리포트", use_container_width=True):
            st.session_state.stage = "report"
            st.rerun()
    with col2:
        if st.button("💬 감정 대화 시작", use_container_width=True):
            st.session_state.stage = "persona"
            st.rerun()

# emotional report
if st.session_state.stage == "report":
    user_ns = st.session_state.user_namespace
    st.title("📊 감정 인식 리포트")
    
    if st.button("⬅️ 돌아가기"):
        st.session_state.stage = "menu"
        st.rerun()

    st.subheader("주간 리포트")
    with st.spinner("📊 리포트 생성 중..."):
        generate_weekly_reports()
    weekly_reports = load_weekly_reports()

    if not weekly_reports:
        st.info("아직 채팅을 일주일동안 안했습니다!")
    else:
        for filename, report in weekly_reports[::-1]:  # show most recent first
            report_name = filename.replace('.json', '').split(f"{user_ns}_report_")[1]
            with st.expander(f"🗓️ {report_name}", expanded=False):
                st.markdown(f"**1. 주요 감정**\n\n{report.get('주요 감정', '데이터 없음')}")
                st.markdown(f"**2. 반복 상황**\n\n{report.get('반복 상황', '데이터 없음')}")
                st.markdown(f"**3. 반복 트리거**\n\n{report.get('반복 트리거', '데이터 없음')}")
                st.markdown(f"**4. 감정 발생 시간대**\n\n{report.get('감정 발생 시간대', '데이터 없음')}")
                st.markdown(f"**5. 행동 패턴**\n\n{report.get('행동 패턴', '데이터 없음')}")
                st.markdown(f"**6. AI의 해석**\n\n{report.get('AI의 해석', '데이터 없음')}")
                st.markdown(f"**7. 추천 루틴**\n\n{report.get('추천 루틴', '데이터 없음')}")

    daily_reports = load_daily_reports()
    st.subheader("일일 리포트")
    if not daily_reports:
        st.info("아직 채팅을 안하셨습니다!")
    else:
        for filename, report in daily_reports:
            report_name = filename.replace('.json', '').split("_chat_")[1]
            with st.expander(f"🗓️ {report_name}", expanded=False):
                st.markdown(f"**1. 주요 감정**\n\n{report.get('주요 감정', '데이터 없음')}")
                st.markdown(f"**2. 반복 상황**\n\n{report.get('반복 상황', '데이터 없음')}")
                st.markdown(f"**3. 반복 트리거**\n\n{report.get('반복 트리거', '데이터 없음')}")
                st.markdown(f"**4. 감정 발생 시간대**\n\n{report.get('감정 발생 시간대', '데이터 없음')}")
                st.markdown(f"**5. 행동 패턴**\n\n{report.get('행동 패턴', '데이터 없음')}")
                st.markdown(f"**6. AI의 해석**\n\n{report.get('AI의 해석', '데이터 없음')}")
                st.markdown(f"**7. 추천 루틴**\n\n{report.get('추천 루틴', '데이터 없음')}")

# persona select
if st.session_state.stage == "persona":
    st.markdown("<h1 style='text-align: center;'>캐릭터 선택창</h1>", unsafe_allow_html=True)

    persona = st.selectbox("캐릭터를 선택하세요: ", ["-", "팩폭이", "기쁨이", "우울이"])

    if st.button("⬅️ 돌아가기"):
        st.session_state.stage = "menu"
        st.rerun()

    if persona != "-":
        st.session_state.persona = persona
        instr_map = {
            "팩폭이": facter_instructions,
            "기쁨이": joy_instructions,
            "우울이": sadness_instructions
        }
        st.session_state.base_prompt += "\n" + instr_map[persona]
        st.session_state.stage = "weather"
        st.rerun()

# choose emotional weather
if st.session_state.stage == "weather":
    st.markdown(
        """
        <style>
        /* target all st.button elements */
        [data-testid="stButton"] > button {
            min-width: 5rem !important;    /* make them wider */
            min-height: 4rem !important;   /* make them taller */
            font-size: 1.3rem !important;  /* bigger text */
            padding: 1rem 1.5rem !important; /* more internal space */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='text-align: center;'>오늘의 감정 날씨는 무엇인가요?</h2>",
        unsafe_allow_html=True
    )

    weather_opts = ["맑음","갬","비","흐림","천둥","눈","바람"]
    outer = st.columns([1,10,2])

    if st.button("⬅️ 돌아가기"):
        st.session_state.stage = "menu"
        st.rerun()

    with outer[1]:
        cols = st.columns(7, gap="large")
        for col, w in zip(cols, weather_opts):
            if col.button(w, key=f"weather_{w}"):
                st.session_state.weather = w
                st.session_state.stage   = "chat"
                st.rerun()

# chatbot
if st.session_state.stage == "chat":
    # st.rerun()
    persona = st.session_state.get("persona", "")
    st.title(f"{persona}")

    if st.button("⬅️ 돌아가기"):
        st.session_state.stage = "menu"
        st.rerun()

    st.sidebar.header("기능")
    st.sidebar.button("로그아웃", on_click=relogin)
    st.sidebar.button("대화 다시 시작하기", on_click=reset_chat)
    st.sidebar.button("감정 일지 생성", on_click=lambda: generate_daily_journal(st.session_state.chat))

    chat_history = load_history(10)

    # start chat
    gemini_api = os.getenv("GEMINI_API_KEY")

    genai.configure(api_key=gemini_api)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash", 
        system_instruction=st.session_state.base_prompt)
    
    st.session_state.chat = model.start_chat(history=chat_history) 
    st.session_state.messages = []

    weather_prompt = f"""
    사용자의 오늘의 감정 날씨는 {st.session_state.weather}입니다.
    사용자가 선택한 {st.session_state.persona} 캐릭터에 맞게 인사말을 하세요.
    """
    response = st.session_state.chat.send_message(weather_prompt)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.text.strip()
    })

    # chat history button
    st.sidebar.button("챗 히스토리 저장", on_click=lambda: save_todays_chat(st.session_state.messages))

    for msg in st.session_state.messages:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

    user_input = st.chat_input("하고 싶은 얘기를 입력해 주세요: ")

    if user_input:
        context = get_context(user_input, namespace=st.session_state.user_namespace)

        if context != []:
            full_message = f"""
            아래는 사용자의 관련된 메타데이터입니다. 키 값은 일기 날짜고, 벨류 값은 그 날의 대화 내용 입니다. 답변에 참고해 주세요:
            {context}

            사용자의 메세지:
            {user_input}
            """
        else:
            full_message = user_input

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        response = st.session_state.chat.send_message(full_message)
        response = response.text.strip()

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").markdown(response)