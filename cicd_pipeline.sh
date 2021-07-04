
echo "sorting of imports"
pipenv run isort --line-width 200 ./

echo "type checking with mypy"
pipenv run mypy ./

# If you like, use code formatting with black
#echo "code formatting with black"
#pipenv run black ./

# running pytest with test coverage, combine coverage results and create report. Clean up afterwards
pipenv run coverage run --module pytest --ignore=deployment # deployment tests require the prediction endpoint to be deployed to each cloud service to be successful, so ignore until the endpoint is permanently available
pipenv run coverage combine
pipenv run coverage report
pipenv run coverage erase

# docker image build
#   If you decided on the type of deployment mode and if it relies on docker images, perform a full image build here to assure the code did not break image creation
#   in the current state, without a specific deployment mode decided yet, the image builds are handled by the shell script 'executables/cloud_setup.sh'
