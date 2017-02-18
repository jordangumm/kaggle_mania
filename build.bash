
USERNAME=$1
PASSWORD=$2

echo $USERNAME
echo $PASSWORD

mkdir -p data/original && cd data/original
kg download -u "$USERNAME" -p "$PASSWORD" -c 'march-machine-learning-mania-2017'
