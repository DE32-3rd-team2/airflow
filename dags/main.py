import os
from datetime import datetime, timedelta, timezone
from textwrap import dedent

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator


with DAG(
    'Team2',
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={
        'depends_on_past': True,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=1)
    },
    description='Team 2 load airflow DAG',
    schedule_interval="@once",
    start_date=datetime(2024, 10, 4),
    catchup=True,
    tags=['team2'],
) as dag:


    def db():
        from airflow.db import select

        sql = """
            SELECT
            num, file_path
            FROM face_age
            WHERE prediction_result IS NULL
            ORDER BY num
            LIMIT 1
            """
        r = select(sql, 1)

        if len(r) > 0:
            return r[0]
        else:
            return None

    task_get_db = PythonVirtualenvOperator(
            task_id="get_db",
            python_callable=db,
            requirements=["git+https://github.com/DE32-3rd-team2/airflow.git@2.1/db"],
            system_site_packages=False
    )
   
    def pred(**context):
        data = context['task_instance'].xcom_pull(task_ids=f'get_db')

        # 모델 원본
        import requests
        from PIL import Image
        from io import BytesIO

        from transformers import ViTFeatureExtractor, ViTForImageClassification

        # r = requests.get('https://github.com/dchen236/FairFace/blob/master/detected_faces/race_Asian_face0.jpg?raw=true')
        # im = Image.open(BytesIO(r.content))

        img_path = data["file_path"]
        img = Image.open(img_path)

        # Init model, transforms
        model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
        transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

        # Transform our image and pass it through the model
        inputs = transforms(im, return_tensors='pt')
        output = model(**inputs)

        # Predicted Class probabilities
        proba = output.logits.softmax(1)

        # Predicted Classes
        preds = proba.argmax(1)

        age_band_mapping = {
            0: '0-2 years',
            1: '3-9 years',
            2: '10-19 years',
            3: '20-29 years',
            4: '30-39 years',
            5: '40-49 years',
            6: '50-59 years',
            7: '60-69 years',
            8: '70+ years'
        }

        # 모든 나이대에 대한 확률 출력
        for i in range(len(proba[0])):
            print(f"{age_band_mapping[i]} 일 확률 ::: {100*proba[0][i]:.3f}%")

        # 예측결과 출력
        print(f"result : {preds.item()}")
        print(f"result : {age_band_mapping[preds.item()]}")

        return preds.item(), proba[0][preds.item()].item()      # 예측결과와 그 확률 반환


    def save(**context):
        # predict task에서 반환한 값을 이어서 사용하도록 하는 코드
        rst, prob = context['task_instance'].xcom_pull(task_ids=f'predict')
        data = context['task_instance'].xcom_pull(task_ids=f'get_db')
        # data = {'num': 1, 'file_path': '/home/ubuntu/images/11a57bb1-55b0-490f-afd9-a077b4c60057.jpeg'} :: dict

        # 한국 시간
        dt=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S')

        # log가 저장되는 위치, dag파일이 저장된 위치의 상위 디렉토리 밑에 logs/prediction/ 에 저장
        log_path = f"{os.path.dirname(os.path.abspath(__file__))}/../logs/prediction/"

        # 저장경로가 없으면 디렉토리 생성
        os.makedirs(log_path, exist_ok=True)

        # log파일 실제 생성, a 옵션=append, 저장되는 정보는 아래 정의된 3가지
        with open(f"{log_path}/pred.log", "a") as f:
            f.write(f"{data['num']},{rst},{dt}\n")

    task_pred = PythonVirtualenvOperator(
        task_id="predict",
        python_callable=pred,
        requirements=["requests", "transformers","pillow","torch"],
        system_site_packages=False,
        venv_cache_path=f"{os.path.dirname(os.path.abspath(__file__))}/../venv/"
    )

    task_save_log = PythonOperator(
        task_id="save.log",
        python_callable=save
    )

    task_end = EmptyOperator(task_id='end')
    task_start = EmptyOperator(task_id='start')

    task_start >> task_get_db >> task_pred >> task_save_log >> task_end
