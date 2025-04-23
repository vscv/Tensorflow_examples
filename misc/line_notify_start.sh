# 2022-01-11
# Send local image to line notify

arg=$1

#token
token='80mM0Ycf8fx7mxyCaCtemRknn2lBaJI2tndF4bo5WA2'

#api
lme='https://notify-api.line.me/api/notify'

# yep, this is how to make a newline in string.
nowtime="${arg}
        $(date)"

curl -X POST -H "Authorization: Bearer ${token}" -F "message=${nowtime}" ${lme}

