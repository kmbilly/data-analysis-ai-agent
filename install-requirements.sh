# after editing top level dependencies in requirements.in
pip-compile requirements.in
pip install -r requirements.txt
