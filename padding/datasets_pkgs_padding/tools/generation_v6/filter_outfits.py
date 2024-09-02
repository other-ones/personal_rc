from consts_v5 import WEARINGS



print('WEARINGS=[')
count=0
for item in WEARINGS:
    if "outfit" not in item:
        print(f'\"{item}\",',end='')
        count+=1
        if (count%5)==0:
            print() 
print(']')


print('OUTFITS=[')
count=0
for item in WEARINGS:
    if "outfit" in item:
        print(f'\"{item}\",',end='')
        count+=1
        if (count%5)==0:
            print() 
print(']')

