import requests
import pandas as pd
import streamlit as st
import datetime
from urllib.parse import urlencode, parse_qs


api_key = "live_50138df357699939f3b790093592e8e075fc0519e008653d27fc175e2dc9da5ad961e05fef3f2d19e258e30bf46403b6"
header = {'x-nxopen-api-key': api_key}

code = st.experimental_get_query_params()
qs = None
st.experimental_set_query_params(**parse_qs(qs))
if "char" in code:
    st.session_state["char"] = code["char"][0]


if "char" not in st.session_state:
    st.session_state["char"] = "마하방패"


char_name = st.session_state["char"]
character_name = st.text_input('캐릭터명을 입력하세요', f"{char_name}")
#ocid 발급
url = f"https://open.api.nexon.com/maplestory/v1/id?character_name={character_name}"
res = requests.get(url, headers= header).json()
if "ocid" in res.keys():
    ocid = res["ocid"]
    
    #기본정보 조회
    s_date = datetime.datetime.today() - datetime.timedelta(days=1)
    s_date = s_date.strftime("%Y-%m-%d")
    url = f"https://open.api.nexon.com/maplestory/v1/character/basic?ocid={ocid}&date={s_date}"
    res = requests.get(url, headers= header).json()

    df = pd.DataFrame.from_dict([res])
    df1 = df[["character_name","character_level", "world_name", "character_gender", "character_class", "character_class_level", "character_guild_name"]] 
    df1 = df1.rename(columns={"character_name":"캐릭명", "character_level":"레벨","world_name":"서버", "character_gender":"성별", "character_class":"직업", "character_class_level":"달성전직", "character_guild_name":"길드명"})
    df2 = df1.set_index("캐릭명")


    c_img = df["character_image"][0]
    st.markdown(f'<img src="{c_img}">', unsafe_allow_html= True)
    st.dataframe(df2)

    # 무릉도장 정보 조회
    url = f"https://open.api.nexon.com/maplestory/v1/character/dojang?ocid={ocid}&date={s_date}"
    res = requests.get(url, headers= header).json()
    best_record = res["dojang_best_floor"]
    best_date = res["date_dojang_record"]
    best_time = res["dojang_best_time"]
    if best_date == None:
        best_date = "기록없음"
    st.markdown(f"# 무릉도장 기록:  {best_record}층 / {best_time}초")
    
    #유니온 정보 조회
    url = f"https://open.api.nexon.com/maplestory/v1/user/union?ocid={ocid}&date={s_date}"
    res = requests.get(url, headers= header).json()
    union_level =res["union_level"]
    union_grade =res["union_grade"]
    st.markdown(f"# {union_grade}/{union_level}레벨")

    #상세 스탯 정보 조회
    url = f'https://open.api.nexon.com/maplestory/v1/character/stat?ocid={ocid}&date={s_date}'
    res = requests.get(url, headers= header).json()
    res = res["final_stat"]
    df = pd.DataFrame(res)
    df = df.rename(columns={"stat_name":"스탯명", "stat_value":"값"})
    cobat_point = df[df["스탯명"]=="전투력"]["값"]
    combat_point = '{:,}'.format(int(cobat_point))
    df = df.set_index("스탯명")
    
    

            
    st.markdown(f"# 전투력 : {combat_point}")
    
    st.dataframe(df, use_container_width=True)

    #어빌리티 조회 및 표시
    url = f'https://open.api.nexon.com/maplestory/v1/character/ability?ocid={ocid}&date={s_date}'
    res = requests.get(url, headers= header).json()
    res = res["ability_info"]
    
    p_grade = {"레전드리":"green", "유니크": "orange", "에픽":"violet", "레어":"blue", "노멀":"white"}
    st.markdown("어빌리티")
    for i in range(len(res)):
        resq = res[i]
        a_grade = resq["ability_grade"]
        abil = resq["ability_value"]
        a_grade_color = p_grade[a_grade]
        st.markdown(f':{a_grade_color}[{abil}]')
    st.divider()
    
    # 착용 아이템 장비 조회
    
    url = f"https://open.api.nexon.com/maplestory/v1/character/item-equipment?ocid={ocid}&date={s_date}"
    res = requests.get(url, headers= header).json()
    res = res["item_equipment"]
    col1, col2, col3,col4 = st.columns(4)
    
    for i in range(len(res)):
        res_i = res[i]
        e_img = res_i["item_icon"]
        e_name = res_i["item_name"]
        e_star = res_i["starforce"]
        e_p_grade = res_i["potential_option_grade"]
        e_a_grade = res_i["additional_potential_option_grade"]
        e_poten1 = res_i["potential_option_1"]
        e_poten2 = res_i["potential_option_2"]
        e_poten3 = res_i["potential_option_3"]
        e_poten4 = res_i["additional_potential_option_1"]
        e_poten5 = res_i["additional_potential_option_2"]
        e_poten6 = res_i["additional_potential_option_3"]
        p_grade = {"레전드리":"green", "유니크": "orange", "에픽":"violet", "레어":"blue", "노멀":"white"}
                
        j = (i%4)+1
        if j == 1:
            j = col1
        if j == 2:
            j = col2
        if j == 3:
            j = col3
        if j == 4:
            j = col4
        with j:
            if e_star =="0":
                e_stare = ""
            else:
                e_stare = e_star+"성"
            st.markdown(f':red[{e_stare}] :gray[{e_name}]<img src="{e_img}">',unsafe_allow_html= True)
            if e_poten1 == None:
                pass
            else:
                p_grade_color = p_grade[e_p_grade]
                st.markdown(f':{p_grade_color}[윗잠: {e_poten1}/{e_poten2}/{e_poten3}]',unsafe_allow_html= True)
            if e_poten4 == None:
                pass
            else:
                a_grade_color = p_grade[e_a_grade]
                st.markdown(f':{a_grade_color}[아랫잠: {e_poten4}/{e_poten5}/{e_poten6}]',unsafe_allow_html= True)
            st.divider()




else:
    st.header(f"{character_name}: 해당 캐릭터의 기록이 없어요")




    
