from datetime import datetime, timedelta
from textwrap import dedent

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonVirtualenvOperator


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
    schedule_interval="* * * * *",
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

    task_end = EmptyOperator(task_id='end')
    task_start = EmptyOperator(task_id='start')

    task_start >> task_get_db >> task_end
