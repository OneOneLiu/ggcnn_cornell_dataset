from scipy.spatial.transform import Rotation as R
import numpy as np
import transforms3d as tfs
#四元数转旋转矩阵
def get_matrix_quate(x, y, z,q1,q2,q3,q4):
    Rq = [q2,q3,q4,q1]
    Rm = R.from_quat(Rq)
    Rm = Rm.as_matrix()
    Rm2 = tfs.affines.compose(np.squeeze(np.asarray((x, y, z))), Rm, [1, 1, 1])
    return Rm2,Rm
 
Pc=[72.323,171.863,1022.452]
Tgb =[850.08,-64.82,760.68,0.48883,-0.50049,0.46997,-0.53822]
Tcg = [184.324,-261.431,182.170,0.998517,0.012349,-0.038843,0.036090]
#四元数-->旋转矩阵（scipy）
qm_cg,Rmcg = get_matrix_quate(Tcg[0], Tcg[1], Tcg[2], Tcg[3], Tcg[4], Tcg[5], Tcg[6])
qm_gb,Rmgb = get_matrix_quate(Tgb[0],Tgb[1],Tgb[2],Tgb[3],Tgb[4],Tgb[5],Tgb[6])
print('位姿四元数转旋转矩阵：\n',qm_gb)
Pc = Pc+[1]
Pc = np.array(Pc)
Pg = np.dot(qm_cg,Pc)
Pb = np.dot(qm_gb,Pg)
print('Pb:',Pb)
