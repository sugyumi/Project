import fisheyeCalibration0724_NODRAW as FC
import math

f = open("valid.txt", 'w', encoding="UTF-8")
for name in FC.valid:
    print(name,file=f)
# print("openCV rms:", FC.rms, file=f)
print("openCV rms:", round(FC.rms, 4), file=f)
# print("Reprojection Error :", np.sqrt(sum(meanErrorTotal) / N_OK), file=f)
print(f'소요시간: {round(FC.end-FC.start, 2)}초', file=f)
for i in range(len(FC.rvecs)):
    for j in range(3):
        rvecs_deg = math.degrees(FC.rvecs[i][j])
        print(FC.rvecs[i][j], round(rvecs_deg, 1), file=f)
    print("", file=f)
f.close()