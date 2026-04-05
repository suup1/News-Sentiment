pipeline {
    agent any

    stages {

        stage('Verify Files') {
            steps {
                bat 'echo === ROOT FILES ==='
                bat 'dir'
                bat 'echo === DATA FOLDER ==='
                bat 'dir data || echo Data folder not present yet'
            }
        }

        stage('Setup Virtual Environment') {
            steps {
                bat '''
                python -m venv venv
                call venv\\Scripts\\activate
                pip install --upgrade pip
                pip install -r requirements.txt
                pip install "dvc[azure]"
                '''
            }
        }

        stage('Pull Data (DVC)') {
    steps {
        withCredentials([string(credentialsId: 'AZURE_KEY', variable: 'AZURE_STORAGE_KEY')]) {
            withEnv([
                "AZURE_STORAGE_ACCOUNT=sentimentanalysis1234"
            ]) {
                bat '''
                call venv\\Scripts\\activate
                echo === PULLING DATA FROM DVC ===
                dvc pull -v
                echo === VERIFY DATA ===
                dir data
                '''
            }
        }
    }
}

        stage('Train Model') {
            steps {
                bat '''
                call venv\\Scripts\\activate
                python src/train.py
                '''
            }
        }

        stage('Archive Model') {
            steps {
                archiveArtifacts artifacts: 'models/*.pkl', fingerprint: true
            }
        }
    }
}