# ML services course project on FastAPI

В этом репозитории проект по курсу создания МЛ сервисов! 

В этом примере решается задача классификации цифр на языке жестов из изображения. Данный код можно также использовать для решения других проблем - в таком случае достаточно изменить содержимое ml составляющей проекта. 

## Local development

```bash
# Create a virtual environment
python3.11 -m venv env

# Activate the virtual environment
source env/bin/activate

# Install/upgrade dependencies
pip install -r requirements.txt

# (Optional) Code formatting
make pretty

# Run tests for ml code
make test_ml

# Run app
make run_app

# Deactivate the virtual environment
deactivate
```

## Run app in docker container

```bash
docker build -t ml-app .
docker run -p 80:80 ml-app
```

## Run tests for the app 

Run the following commands while docker container is running (in other terminal).

```bash
source env/bin/activate
make test_app

deactivate
```