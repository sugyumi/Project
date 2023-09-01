import cv2
import numpy as np
import glob
import time

start = time.time()
# 체커보드 내부 코너 개수
CHECKERBOARD = (9, 10)
checkboard_size = CHECKERBOARD[0] * CHECKERBOARD[1]

# 체커보드의 한 변의 길이 (mm)
square_size = 60

# 객체 포인트 생성
objp = np.zeros((1, checkboard_size, 3), np.float64)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

# 모든 이미지에서 객체 포인트와 이미지 포인트를 저장하기 위한 배열
objpoints = []  # 3D 공간 상의 좌표 (객체 포인트)
imgpoints = []  # 2D 이미지 상의 좌표 (이미지 포인트)

# 이미지 파일 한 번에 불러오기
images = glob.glob('0901_0004/1/*.bmp')
print(images)

# 첫 번째 이미지 형태(크기) 저장용 변수 초기화
# 이 변수를 사용해 이미지 별 크기 비교 후 불일치 시 에러 표시.
_img_shape = None
validImgs = []

for fname in images:
    # 모든 이미지의 크기가 동일한 지 확인 후 회색조로 변환
    img = cv2.imread(fname)
    if _img_shape is None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체커보드의 코너점 찾기
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH  # 코너 검출 시 이미지 각 지역마다 다른 임계값 사용해 코너 검출
        + cv2.CALIB_CB_FAST_CHECK  # 코너 검출 속도 향상
        + cv2.CALIB_CB_NORMALIZE_IMAGE  # 이미지 밝기 정규화 후 코너 검출 > 밝기 불균일성 보정, 정확도 향상
    )

    # 코너 검출에 성공 시, objpoints 배열과 imgpoints 배열에 각각 추가
    if ret:
        validImgs.append(fname)
        # 코너 좌표 확인 및 출력
        # for i in range(len(corners)):
        #     corner = corners[i].ravel()
        #     print("코너 {}의 좌표: ({}, {})".format(i + 1, corner[0], corner[1]))

        # subpix_criteria: 코너 서브픽셀 검출에 사용되는 기준값을 나타내는 변수
        # 알고리즘 최대 30번 반복(MAX_ITER), 반복 정확도 0.1보다 작아지면 알고리즘 종료(EPS)
        # 반복 정확도: 코너 위치가 0.1 보다 작게 이동하는 경우
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

        # cv2.cornerSubPix(image, corners, winsize, zeroZone, criteria)
        # winsize : Half of the side length of the search window(searches for corners)
        # zeroZone : This is the half of the neighborhood size want to reject. The value of (-1,-1) indicates that there is no such a size.
        # findChessboardCorners로 찾은 각 코너점의 위치를 서브픽셀 수준으로 다시 계산하여 반환
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)  # 코너 보정

        objpoints.append(objp)  # 체커보드의 각 코너에 대한 3D 좌표(objp) 추가
        imgpoints.append(corners)  # 보정된 코너 좌표(corners, 이미지 평면상 2D 좌표) 추가

# Calibration에 필요한 변수 초기화
N_OK = len(objpoints)  # objpoints: 검출된 코너에 대한 3D 좌표 / N_OK: 코너가 성공적으로 검출된 이미지의 개수
K = np.zeros((3, 3))  # Intrinsic Parameter Matrix
D = np.zeros((4, 1))  # Distortion Coefficient
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]  # 회전 벡터
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]  # 변위 벡터

# calibration_flags: fisheye.calibrate 함수에서 인자로 사용되는 플래그(flag)의 조합
# 외부 매개변수 다시 계산(RE_EX), 보정 조건 유효성 검사(CHE_COND), 왜곡 고정(FIX_SKEW)
calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        # + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
)

calibration_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(  # rms, K, D, rvecs, tvecs 반환
    objpoints,  # 3D 공간 상의 좌표
    imgpoints,  # 2D 이미지 상의 좌표
    gray.shape[::-1],  # 이미지 크기(1600x1300)
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    calibration_criteria
)

end = time.time()

meanErrorTotal = []
for idx in range(N_OK):
    count = 0
    error = 0
    for i in range(checkboard_size):
        # 코너의 좌표
        corner_index = count
        corner = imgpoints[idx][corner_index].ravel() # 1차원으로 변경

        # 코너의 좌표에서 reprojection 좌표 계산
        re_imgpoints, _ = cv2.fisheye.projectPoints(objp, rvecs[idx], tvecs[idx], K, D)  # K,D 는 1개값 r, t 는 이미지 개수만큼
        corner_reprojection = re_imgpoints[0][corner_index]

        # 코너가 float32 형태이므로 float64 로 변환(corner_reprojection: float64)
        corner = corner.astype(np.float64)

        # 코너의 reprojection error 계산 @@@
        # NORM_L2SQR: ∑(corner - corner_reprojection)², 정규화 방법 중 하나
        error += cv2.norm(corner, corner_reprojection, cv2.NORM_L2SQR)

        count += 1

    # 평균 reprojection error 계산
    mean_error = error / checkboard_size
    meanErrorTotal.append(mean_error)

    # ravel된 re_imgpoints를 3차원으로 변환
    re_imgpoints = re_imgpoints.reshape(checkboard_size, 1, 2)

    # 그리기
    circle_img = cv2.imread(validImgs[idx])

    for j in range(checkboard_size):
        cv2.circle(circle_img, (int(imgpoints[idx][j][0][0]), int(imgpoints[idx][j][0][1])),
                   6, (255, 0, 255), -1)  # 2d points (pink)
        cv2.circle(circle_img, (int(re_imgpoints[j][0][0]), int(re_imgpoints[j][0][1])),
                   6, (0, 255, 0), -1)  # reprojected 3d points (green)

    # cv2.imwrite(f'images{idx}.jpg', circle_img)  # 이미지 저장용
    resized_img = cv2.resize(circle_img, None, fx=0.7, fy=0.7)
    cv2.imshow(f'{validImgs[idx]}', resized_img)
    cv2.waitKey(0)

# print("K:\n", K)
# print("D:\n", D)
# print("rvecs:\n", rvecs)
# print("tvecs:\n", tvecs)
# print("openCV rms:", round(rms, 4))
print("openCV rms:", rms)
# print("Reprojection Error :", round(np.sqrt(sum(meanErrorTotal) / N_OK), 4))
# print("Reprojection Error :", np.sqrt(sum(meanErrorTotal) / N_OK))
# print("openCV rms == our Reprojection Error?", rms == np.sqrt(sum(meanErrorTotal) / N_OK))
print(f'Found {N_OK} valid images for calibration')
print(f'valid images: {validImgs}')
print(f'소요시간:d {end-start}')

f = open("valid.txt", 'w')
for name in validImgs:
    print(name, file=f)
print("K\n", K, file=f)
print("D\n", D, file=f)
print("openCV rms:", rms, file=f)
# print("Reprojection Error :", np.sqrt(sum(meanErrorTotal) / N_OK), file=f)
f.close()
