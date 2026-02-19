#!/bin/bash
# Script to restart celery/shop-workers and update ELK container mapping

# Ensure we are in the project root
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_ROOT"

echo "Restarting Docker Compose profiles: celery, shop-workers..."
docker compose --profile celery --profile shop-workers restart

echo "Updating ELK container mapping..."
# Executing the script as requested by the user
bash /home/samuelzhu/ELK/logstash/update-container-mapping.sh

echo "Done."
