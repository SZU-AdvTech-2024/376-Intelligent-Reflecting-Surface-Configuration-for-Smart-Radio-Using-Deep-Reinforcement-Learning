"""
多用户MIMO环境
"""

from __future__ import division
import numpy as np
import math

from math import *


class envMuMIMO:

    def __init__(self, NumAntBS, NumEleIRS, NumUser):
        """
        初始化多用户MIMO环境

        :param NumAntBS: 基站天线数量
        :param NumEleIRS: IRS（智能反射面）单元数量
        :param NumUser: 用户数量
        """
        self.NumAntBS = NumAntBS
        self.NumEleIRS = NumEleIRS
        self.NumUser = NumUser

    def DFT_matrix(self, N_point):
        """
        生成DFT矩阵
        :param N_point: 点数
        :return: N_point x N_point 的DFT矩阵
        """
        n, m = np.meshgrid(np.arange(N_point), np.arange(N_point))
        omega = np.exp(-2 * pi * 1j / N_point)
        W = np.power(omega, n * m) / sqrt(N_point)  # 归一化DFT矩阵
        return W

    def CH_Prop(self, H, sigma2, Pilot):
        """
        信道传播模型

        :param H: 信道矩阵 [NumAntBS, NumUser]
        :param sigma2: 噪声方差
        :param Pilot: 导频信号 [NumUser, 1]
        :return: 接收信号 [NumAntBS, 1]
        """
        NumAnt, NumUser = np.shape(H)
        noise = 1 / sqrt(2) * (np.random.normal(0, sigma2, size=(NumAnt, NumUser))
                               + 1j * np.random.normal(0, sigma2, size=(NumAnt, NumUser)))  # 高斯噪声
        y_rx = np.dot(H, Pilot) + noise  # 接收信号 = 信道 * 导频 + 噪声
        return y_rx

    def CH_est(self, y_rx, sigma2, Pilot):
        """
        信道估计

        :param y_rx: 接收信号 [NumAntBS, 1]
        :param sigma2: 噪声方差
        :param Pilot: 导频信号 [NumUser, 1]
        :return: 估计的信道矩阵 [NumAntBS, NumUser]
        """
        MMSE_matrix = np.matrix.getH(Pilot) / (1 + sigma2)  # MMSE信道估计矩阵
        H_est = np.dot(y_rx, MMSE_matrix)  # 估计的信道
        return H_est

    def Precoding(self, H_est):
        """
        预编码

        :param H_est: 估计的信道矩阵 [NumAntBS, NumUser]
        :return: 预编码矩阵 [NumAntBS, NumUser]
        """
        F = np.dot(np.linalg.inv(np.dot(np.matrix.getH(H_est), H_est)),
                   np.matrix.getH(H_est))  # 零强制预编码
        NormCoeff = abs(np.diag(np.dot(F, np.matrix.getH(F))))  # 计算归一化系数
        NormCoeff = 1 / np.sqrt(NormCoeff)
        F = np.dot(np.diag(NormCoeff), F)  # 归一化预编码矩阵
        return F

    def GetRewards(self, Pilot, H_synt, sigma2_BS, sigma2_UE):
        """
        获取奖励

        :param Pilot: 导频信号 [NumUser, 1]
        :param H_synt: 综合无线信道 [NumAntBS, NumUser]
        :param sigma2_BS: 基站噪声方差
        :param sigma2_UE: 用户端噪声方差
        :return: 数据速率 [NumUser, 1], 接收信号 [NumAntBS, 1], 估计的等效信道 [NumAntBS, NumUser]
        """
        y_rx = self.CH_Prop(H_synt, sigma2_BS, Pilot)  # 接收信号
        H_est = self.CH_est(y_rx, sigma2_BS, Pilot)  # 估计等效信道
        F = self.Precoding(H_est)  # 零强制预编码
        H_eq = np.dot(F, H_synt)  # 等效信道 [NumAntBS, NumUser]->[NumAntBS, NumUser]
        H_eq2 = abs(H_eq * np.conj(H_eq))  # 信道功率
        SigPower = np.diag(H_eq2)  # 信号功率
        IntfPower = H_eq2.sum(axis=0)  # 干扰功率
        IntfPower = IntfPower - SigPower  # 减去信号功率得到干扰功率
        SINR = SigPower / (IntfPower + sigma2_UE)  # 计算SINR
        Rate = np.log2(1 + SINR)  # 计算数据速率
        return Rate, y_rx, H_est

    def SubSteeringVec(self, Angle, NumAnt):
        """
        子波束导向向量

        :param Angle: 角度
        :param NumAnt: 天线数量
        :return: 子波束导向向量 [NumAnt, 1]
        """
        SSV = np.exp(1j * Angle * math.pi * np.arange(0, NumAnt, 1))
        SSV = SSV.reshape(-1, 1)  # 转换为列向量
        return SSV

    def ChannelResponse(self, Pos_A, Pos_B, ArrayShape_A, ArrayShape_B):
        """
        计算位置相关的LoS信道响应

        :param Pos_A: 位置A
        :param Pos_B: 位置B
        :param ArrayShape_A: A处天线阵列形状
        :param ArrayShape_B: B处天线阵列形状
        :return: 信道响应矩阵 [NumAntA, NumAntB]
        """
        dis_AB = np.linalg.norm(Pos_A - Pos_B)  # 计算距离
        DirVec_AB = (Pos_A - Pos_B) / dis_AB  # 方向向量
        angleA = [np.linalg.multi_dot([[1, 0, 0], DirVec_AB]), np.linalg.multi_dot([[0, 1, 0], DirVec_AB]),
                  np.linalg.multi_dot([[0, 0, 1], DirVec_AB])]  # A处的角度
        SteeringVectorA = np.kron(self.SubSteeringVec(angleA[0], ArrayShape_A[0]),
                                  self.SubSteeringVec(angleA[1], ArrayShape_A[1]))
        SteeringVectorA = np.kron(SteeringVectorA, self.SubSteeringVec(angleA[2], ArrayShape_A[2])) # A处的导向向量
        angleB = [np.linalg.multi_dot([[1, 0, 0], DirVec_AB]), np.linalg.multi_dot([[0, 1, 0], DirVec_AB]),
                  np.linalg.multi_dot([[0, 0, 1], DirVec_AB])]  # B处的角度
        SteeringVectorB = np.kron(self.SubSteeringVec(angleB[0], ArrayShape_B[0]),
                                  self.SubSteeringVec(angleB[1], ArrayShape_B[1]))
        SteeringVectorB = np.kron(SteeringVectorB, self.SubSteeringVec(angleB[2], ArrayShape_B[2])) # B处的导向向量
        H_matrix = np.linalg.multi_dot([SteeringVectorA, np.matrix.getH(SteeringVectorB)])  # 计算信道响应
        return H_matrix

    def H_GenFunLoS(self, Pos_BS, Pos_IRS, Pos_UE, ArrayShape_BS, ArrayShape_IRS, ArrayShape_UE):
        """
        生成LoS信道

        :param Pos_BS: 基站位置
        :param Pos_IRS: IRS位置
        :param Pos_UE: 用户位置列表
        :param ArrayShape_BS: 基站天线阵列形状
        :param ArrayShape_IRS: IRS天线阵列形状
        :param ArrayShape_UE: 用户天线阵列形状
        :return: BS到UE的信道 [NumAntBS, NumUE], BS到IRS的信道 [NumAntBS, NumEleIRS], IRS到UE的信道 [NumEleIRS, NumUE]
        """
        NumUE = len(Pos_UE)
        NumAntBS = np.prod(ArrayShape_BS)  # 基站天线总数
        NumEleIRS = np.prod(ArrayShape_IRS)  # IRS单元总数
        H_BU_LoS = np.zeros((NumAntBS, NumUE)) + 1j * np.zeros((NumAntBS, NumUE))  # 初始化BS到UE的信道
        H_RU_LoS = np.zeros((NumEleIRS, NumUE)) + 1j * np.zeros((NumEleIRS, NumUE))  # 初始化IRS到UE的信道
        for iu in range(NumUE):
            h_BU_LoS = self.ChannelResponse(Pos_BS, Pos_UE[iu], ArrayShape_BS, ArrayShape_UE)  # 计算单个用户的信道
            H_BU_LoS[:, iu] = h_BU_LoS.reshape(-1)  # 将单个用户的信道添加到总信道中
            h_RU_LoS = self.ChannelResponse(Pos_IRS, Pos_UE[iu], ArrayShape_IRS, ArrayShape_UE)  # 计算单个用户的信道
            H_RU_LoS[:, iu] = h_RU_LoS.reshape(-1)  # 将单个用户的信道添加到总信道中
        H_BR_LoS = self.ChannelResponse(Pos_BS, Pos_IRS, ArrayShape_BS, ArrayShape_IRS)  # 计算BS到IRS的信道
        return H_BU_LoS, H_BR_LoS, H_RU_LoS

    def H_GenFunNLoS(self, NumAntBS, NumEleIRS, NumUser):
        """
        生成NLoS信道

        :param NumAntBS: 基站天线数量
        :param NumEleIRS: IRS单元数量
        :param NumUser: 用户数量
        :return: UE到BS的NLoS信道 [NumAntBS, NumUser], BS到IRS的NLoS信道 [NumAntBS, NumEleIRS], UE到IRS的NLoS信道 [NumEleIRS, NumUser]
        """
        H_U2B_NLoS = 1 / sqrt(2) * (np.random.normal(0, 1, size=(NumAntBS, NumUser))
                                    + 1j * np.random.normal(0, 1, size=(NumAntBS, NumUser)))  # 生成复高斯随机变量
        H_R2B_NLoS = 1 / sqrt(2) * (np.random.normal(0, 1, size=(NumAntBS, NumEleIRS))
                                    + 1j * np.random.normal(0, 1, size=(NumAntBS, NumEleIRS)))  # 生成复高斯随机变量
        H_U2R_NLoS = 1 / sqrt(2) * (np.random.normal(0, 1, size=(NumEleIRS, NumUser))
                                    + 1j * np.random.normal(0, 1, size=(NumEleIRS, NumUser)))  # 生成复高斯随机变量
        return H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS

    def H_syntFun(self, H_U2B, H_R2B, H_U2R, RefVector):
        """
        综合无线信道

        :param H_U2B: UE到BS的信道 [NumAntBS, NumUser]
        :param H_R2B: IRS到BS的信道 [NumAntBS, NumEleIRS]
        :param H_U2R: UE到IRS的信道 [NumEleIRS, NumUser]
        :param RefVector: 反射矢量 [NumEleIRS, 1]
        :return: 综合无线信道 [NumAntBS, NumUser]
        """
        RefPattern_matrix = np.diag(RefVector)  # 反射模式矩阵
        H_synt = H_U2B + 1 * np.linalg.multi_dot([H_R2B, RefPattern_matrix, H_U2R])  # 计算综合无线信道, 1 为经过IRS反射后的权重
        return H_synt
