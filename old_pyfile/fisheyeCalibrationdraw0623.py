import cv2
import numpy as np
import os
import glob

# 체커보드 내부 코너 개수

# subpix_criteria: 코너 서브픽셀 검출에 사용되는 기준값을 나타내는 변수
# 알고리즘 최대 30번 반복(MAX_ITER), 반복 정확도 0.1보다 작아지면 알고리즘 종료(EPS)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
CHECKERBOARD = (6, 7)

# 체스보드의 한 변의 길이 (mm)
square_size = 25

# calibration_flags: OpenCV Calibration 과정에서 사용되는 플래그(flag)의 조합을 정의
# 외부 매개변수 다시 계산(RE_EX), 보정 조건 유효성 검사(CHE_COND), 왜곡 고정(FIX_SKEW)
calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
)

# 객체 포인트 생성
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float64)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

# 첫 번째 이미지 형태(크기) 저장용 변수 초기화
# 모든 이미지는 동일한 크기여야 하므로 이 변수를 사용해 크기 비교 후 불일치 시 에러 표시.
_img_shape = None

# 모든 이미지에서 객체 포인트와 이미지 포인트를 저장하기 위한 배열
objpoints = []  # 3D point in real world space
imgpoints = []  # 2D points in image plane

# 이미지 파일 한 번에 불러오기
images = glob.glob('./0622_fisheye/*.bmp')

for fname in images:
    # 모든 이미지의 크기가 동일한 지 확인 후 회색조로 변환
    img = cv2.imread(fname)
    if _img_shape is None:
        _img_shape = img.shape[:2]
    else:
        # assert 조건, "에러 메세지" >> 조건문 미충족 시 에러 메세지 출력, 프로그램 중단
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH  # 코너 검출 시 이미지 각 지역마다 다른 임계값 사용해 코너 검출
        + cv2.CALIB_CB_FAST_CHECK  # 코너 검출 속도 향상 (주석처리해도 값은 똑같이 나옴)
        + cv2.CALIB_CB_NORMALIZE_IMAGE  # 이미지 밝기 정규화 후 코너 검출 > 밝기 불균일성 보정, 정확도 향상
    )

    # If found, add object points, image points (after refining them)

    if ret:  # 코너 검출에 성공했을 시
        objpoints.append(objp)  # 체스보드의 각 코너에 대한 3D 좌표(objp) 추가
        imgpoints.append(corners)  # 보정된 코너 좌표(corners, 이미지 평면상 2D 좌표) 추가
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)  # 코너 보정


# Calibration에 필요한 변수 초기화
N_OK = len(objpoints)  # objpoints: 검출된 코너에 대한 3D 좌표 >> N_OK: 코너가 성공적으로 검출된 이미지의 개수
K = np.zeros((3, 3))  # Intrinsic Parameter Matrix
D = np.zeros((4, 1))  # Distortion Coefficient
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]  # 회전 벡터
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]  # 변위 벡터

rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(  # rms, K, D, rvecs, tvec 반환.(그 중 rms값만 사용. 필요 시 변수 선언하면 될 듯?)
    objpoints,  # 3D 공간 상의 좌표
    imgpoints,  # 2D 이미지 상의 좌표
    gray.shape[::-1],  # 이미지 크기
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
)

# print("openCV:", rms)
# print("K:", K)
# print("D:", D)
# print("rvecs:", rvecs[0])
# print("tvecs:", tvecs[0])


meanErrortotal = []
for idx in range(N_OK):
    count = 0
    for i in range(CHECKERBOARD[0] * CHECKERBOARD[1]):
        # 코너의 좌표
        corner_index = count
        corner = imgpoints[idx][corner_index].ravel()
        # corner = corners[corner_index].ravel()
        # print(len(corner))

        # 코너의 좌표에서 reprojection 좌표 계산
        imgpoints2, _ = cv2.fisheye.projectPoints(objp, rvecs[idx], tvecs[idx], K, D)  # K,D 는 1개값 r, t 는 이미지개수만큼..
        corner_reprojection = imgpoints2[0][corner_index]

        # print(len(corner_reprojection))
        # print(imgpoints2[0][0][i])

        # print("===========")
        corner = corner.astype(np.float64)
        # print(corner.dtype, corner_reprojection.dtype)

        # 코너의 reprojection error 계산
        error = cv2.norm(corner, corner_reprojection, cv2.NORM_L2) / len(corner_reprojection)
        # print("코너 {}의 reprojection error: {}".format(corner_index+1, error))
        error += error

        count += 1

    # 평균 reprojection error 계산
    mean_error = error / CHECKERBOARD[0] * CHECKERBOARD[1]  # ()괄호 묶고 안묶고 차이가 있네
    meanErrortotal.append(mean_error)
    # print(f'{idx} 번째 reprojection error: {round(mean_error, 4)}')
    # print("\n평균 reprojection error:")
    # print(mean_error)

    imgpoints2 = imgpoints2.reshape(42, 1, 2)

    # 그리기
    img = cv2.imread(images[idx])

    # cv2.drawChessboardCorners(img, CHECKERBOARD, imgpoints2, ret)
    # cv2.drawChessboardCorners(img, CHECKERBOARD, imgpoints[idx], ret)

    for j in range(CHECKERBOARD[0] * CHECKERBOARD[1]):
        cv2.circle(img, (int(imgpoints2[j][0][0]), int(imgpoints2[j][0][1])), 3, (0,255,0), 2)
        cv2.circle(img, (int(imgpoints[idx][j][0][0]), int(imgpoints[idx][j][0][1])), 1, (255,0,255), 2)
    cv2.imshow('Corners', img)
    cv2.waitKey(10)


# for i in range(len(objpoints)):
#     imgpoints_reproj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
    # 좌표 찍어보기
    # 다른 왜곡 계수 여부 확인
    # error = cv2.norm(imgpoints[i], imgpoints_reproj, cv2.NORM_L2) / len(imgpoints_reproj)
    # mean_error += error

    # img = cv2.imread(images[i])
    # cv2.drawChessboardCorners(img, CHECKERBOARD, imgpoints_reproj, ret)
    # cv2.drawChessboardCorners(img, CHECKERBOARD, imgpoints[i], ret)
    # cv2.imshow('Corners', img)
    # cv2.waitKey(100)

# 평균 Reprojection 오차 계산 및 출력
# mean_error /= len(objpoints)
# print("평균 Reprojection 오차:", mean_error)

print("Reprojection Error : ", end="")
print(round(sum(meanErrortotal) / N_OK, 4))

print("Found " + str(N_OK) + " valid images for calibration")
# print("DIM=" + str(_img_shape[::-1]))
print()
print("K=np.array(" + str(K.tolist()) + ")")  # Intrinsic Camera Matrix (K)
print("D=np.array(" + str(D.tolist()) + ")")  # Distortion Coefficients (D)
# print(rvecs)
# print(tvecs)
# print("Reprojection Error (RMS):", round(rms, 4))
