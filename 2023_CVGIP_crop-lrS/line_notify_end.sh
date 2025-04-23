# 2022-01-11
# Send local image to line notify

#token
token='80mM0Ycf8fx7mxyCaCtemRknn2lBaJI2tndF4bo5WA2'

#api
lme='https://notify-api.line.me/api/notify'

nowtime="Job Done
        $(date)"

message='Img notify'

curl -X POST -H "Authorization: Bearer ${token}" -F "message=${nowtime}" -F imageFile=@tf.100x100_dark.jpg ${lme}

