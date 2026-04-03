import os
import sys
import random
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, ttest_1samp, pearsonr, spearmanr
from scipy.integrate import nquad, quad, dblquad, simpson
from numpy import trapz
from scipy.optimize import fsolve, minimize, Bounds
from scipy.interpolate import interp1d
from sklearn.feature_selection import mutual_info_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import Symbol, integrate
from tqdm import tqdm
from PIL import Image
try:
    import scienceplots
    plt.style.use(['science', 'ieee'])
except Exception:
    plt.style.use('default')
import cProfile
import pstats
import functools
import argparse
profiler = cProfile.Profile()
profiler.enable()

if len(sys.argv) == 1:
    sys.argv.extend([
        '--alvo_F1=421',
        '--a_F1=250',
        '--b_F1=543',
        '--a_F2=908',
        '--b_F2=1987',
        '--limiar_1=600',
        '--limiar_2=345',
        '--L=1',
        '--k_1=1',
        '--k_2=7',
        '--alvo_F1=421',
        '--alvo_F2=1887',
        '--neutro_F1=610',
        '--neutro_F2=1900',
        '--lambda_zero=1.027',
        '--lambda_RP=0.417',
        '--lambda_RA=0.018',
        '--vogais=e',
        '--caminho_do_arquivo=data.txt'
    ])


def parse_parametros():
    parser = argparse.ArgumentParser()
    parser.add_argument('--otimizar', action='store_true', help="Enable optimization if this flag is set")
    parser.add_argument('--alvo_F1', type=int, default=421)
    parser.add_argument('--alvo_F2', type=int, default=1887)
    parser.add_argument('--limiar_1', type=int, default=600)
    parser.add_argument('--limiar_2', type=int, default=345)
    parser.add_argument('--neutro_F1', type=int, default=610)
    parser.add_argument('--neutro_F2', type=int, default=1900)
    parser.add_argument('--L', type=int, default=1)
    parser.add_argument('--k_1', type=int, default=1)
    parser.add_argument('--k_2', type=int, default=7)
    parser.add_argument('--a_F1', type=int, default=250)
    parser.add_argument('--b_F1', type=int, default=543)
    parser.add_argument('--a_F2', type=int, default=908)
    parser.add_argument('--b_F2', type=int, default=1987)
    parser.add_argument('--lambda_zero', type=float, default=1.027)
    parser.add_argument('--lambda_RA', type=float, default=0.018)
    parser.add_argument('--lambda_RP', type=float, default=0.417)
    parser.add_argument('--vogais', type=str, default="e")
    parser.add_argument('--entrevistados', type=str, default="1,3,5")
    parser.add_argument('--caminho_do_arquivo', type=str, default="data.txt")
    args = parser.parse_args()
    params_todos = vars(args)

    args.vogais = args.vogais.split(',')
    args.entrevistados = list(map(int, args.entrevistados.split(',')))

    dados_keys = ['alvo_F1', 'alvo_F2', 'limiar_1', 'limiar_2', 'neutro_F1', 'neutro_F2',
                  'L', 'k_1', 'k_2', 'a_F1', 'b_F1', 'a_F2', 'b_F2']

    pesos_keys = ['lambda_zero', 'lambda_RA', 'lambda_RP']

    otimizar = params_todos['otimizar']
    params_arq = {'entrevistados': args.entrevistados, 'vogais': args.vogais, 'caminho_do_arquivo': args.caminho_do_arquivo}
    params_dados = {k: params_todos[k] for k in dados_keys}
    params_pesos = {k: params_todos[k] for k in pesos_keys}

    return params_dados, params_pesos, otimizar, params_arq

def definir_diretorio():
    if getattr(sys, 'frozen', False):
        os.chdir(sys._MEIPASS)
    else:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

definir_diretorio()

def ler_dados(caminho_do_arquivo, vogais, entrevistados):
    dados = pd.read_csv(caminho_do_arquivo, sep=" ", quotechar='"', header=0)
    dados = dados[(dados['F1'] != 'NA') & (dados['F2'] != 'NA')]
    dados['F1'] = pd.to_numeric(dados['F1'])
    dados['F2'] = pd.to_numeric(dados['F2'])
    candidatos = dados[dados["Vogal"].isin(vogais) & dados["Falante"].isin(entrevistados)].copy()
    return candidatos[['F1', 'F2']]

def calcular_kde(dados, largura=0.2):
    print('\nCalculating data p.d.f. values...')
    scaler = StandardScaler()
    valores = np.vstack([dados['F1'], dados['F2']]).T
    valores_normalizados = scaler.fit_transform(valores)
    kde = gaussian_kde(valores_normalizados.T, bw_method=largura)
    limits = [(-np.inf, np.inf), (-np.inf, np.inf)]
    integral, error = nquad(lambda x, y: kde(np.vstack([x, y])), limits)
    print(f"\nChecking p.d.f. integral: {integral}")
    if not np.isclose(integral, 1, atol=1e-3):
        print('   |')
        print("   |--> KDE values are not normalized.")
    else:
        print('   |')
        print("   |--> KDE values are normalized.")
    return kde, scaler

def criar_fdp_marginal(kde, params):
    min_F1 = params['min_F1']
    max_F1 = params['max_F1']
    min_F2 = params['min_F2']
    max_F2 = params['max_F2']
    resolucao = params['resolucao']
    grade_F1 = np.linspace(min_F1, max_F1, resolucao)
    grade_F2 = np.linspace(min_F2, max_F2, resolucao)
    F1_mesh, F2_mesh = np.meshgrid(grade_F1, grade_F2)
    pontos_grade = np.vstack([F1_mesh.ravel(), F2_mesh.ravel()])
    valores_kde = kde(pontos_grade).reshape(F1_mesh.shape)
    dx = (max_F1 - min_F1) / (resolucao - 1)
    dy = (max_F2 - min_F2) / (resolucao - 1)
    valores_marginalizados_F1 = np.sum(valores_kde, axis=1) * dy
    integral_marginal_F1 = np.sum(valores_marginalizados_F1) * dx
    if integral_marginal_F1 != 1:
        valores_marginalizados_F1 /= integral_marginal_F1
    funcao_interpolada = interp1d(grade_F1, valores_marginalizados_F1, kind='cubic', fill_value="extrapolate")
    return funcao_interpolada

def arquivo_EDK(kde, params, degrau=0.5):
    limite_F1 = (params['min_F1'], params['max_F1'])
    limite_F2 = (params['min_F2'], params['max_F2'])
    with open('sample_values.txt', 'w') as file:
        file.write("F1, F2, Valor EDK\n")
        for F1 in np.arange(limite_F1[0], limite_F1[1], degrau):
            for F2 in np.arange(limite_F2[0], limite_F2[1], degrau):
                valores = [F1, F2]
                valores_kde = kde(valores)[0]
                file.write(f"{F1}, {F2}, {valores_kde}\n")

def integral_fdp(kde, params):
    fator_superior = 1.5
    fator_inferior = 1
    min_F1, max_F1 = params['min_F1'] * fator_inferior, params['max_F1'] * fator_superior
    min_F2, max_F2 = params['min_F2'] * fator_inferior, params['max_F2'] * fator_superior
    def integrando(F1, F2):
        return kde(np.array([F1, F2]))
    integral, _ = nquad(integrando, [[min_F1, max_F1], [min_F2, max_F2]])
    return integral

def verificacao_kde(kde, params, tolerancia=0.02):
    integral = integral_fdp(kde, params)
    return np.isclose(integral, 1.0, atol=tolerancia)

def restricao_articulatoria(F1, F2, kde, params):
    alvo_F1 = params['alvo_F1']
    alvo_F2 = params['alvo_F2']
    neutro_F1 = params['neutro_F1']
    neutro_F2 = params['neutro_F2']
    esforco_alvo = np.sqrt((alvo_F1 - neutro_F1) ** 2 + (alvo_F2 - neutro_F2) ** 2)
    esforco_producao = np.sqrt((F1 - neutro_F1) ** 2 + (F2 - neutro_F2) ** 2)
    distancia = np.sqrt((F1 - alvo_F1) ** 2 + (F2 - alvo_F2) ** 2)
    dif_esforco = (esforco_producao + 1e-6) / (esforco_alvo + 1e-6)
    RA = distancia * dif_esforco
    return RA

def integral_ra(kde, params):
    a_F1 = params['a_F1']
    b_F1 = params['b_F1']
    a_F2 = params['a_F2']
    b_F2 = params['b_F2']
    funcao_ra = lambda F2, F1: np.exp(restricao_articulatoria(F1, F2, kde, params) - kde([F1, F2])[0])
    integral, erro = dblquad(funcao_ra, a_F2, b_F2, lambda F2: a_F1, lambda F2: b_F1)
    return integral

def restricao_perceptual(F1, kde, params):
    limiar_1 = params['limiar_1']
    limiar_2 = params['limiar_2']
    L = params['L']
    k_1 = params['k_1']
    k_2 = params['k_2']
    produto =  (L**2 / ( (1 + np.exp(k_1 * (F1 - limiar_1))) * (1 + np.exp(-k_2 * (F1 - limiar_2))) ) )
    RP = (L - produto)
    return RP

def integral_rp(kde, params, fdp_marginalizada):
    a_F1 = params['a_F1']
    b_F1 = params['b_F1']
    min_F2 = params['a_F2']
    max_F2 = params['b_F2']
    integral, erro = quad(lambda F1: np.exp(restricao_perceptual(F1, kde, params) - fdp_marginalizada(F1)) , a_F1, b_F1)
    return integral

def escore_estabilidade(kde, scaler, fdp_marginalizada, params, lambda_RA, lambda_RP):
    a_F1 = params['a_F1']
    b_F1 = params['b_F1']
    a_F2 = params['a_F2']
    b_F2 = params['b_F2']
    volume = (b_F1 - a_F1) * (b_F2 - a_F2)
    def funcao_harmonia(F1, F2):
        RA = np.exp(restricao_articulatoria(F1, F2, kde, params) - kde([F1, F2])[0])
        RP = np.exp(restricao_perceptual(F1, kde, params) - fdp_marginalizada(F1))
        return lambda_RA * RA + lambda_RP * RP
    def derivada_funcao_harmonia(F1, F2):
        def derivada_F1(F1, F2):
            h = 1e-5
            return (funcao_harmonia(F1 + h, F2) - funcao_harmonia(F1, F2)) / h
        def derivada_F2(F1, F2):
            h = 1e-5
            return (funcao_harmonia(F1, F2 + h) - funcao_harmonia(F1, F2)) / h
        return (derivada_F1(F1, F2) + derivada_F2(F1, F2))
    integral, erro = dblquad(derivada_funcao_harmonia, a_F1, b_F1, lambda F1: a_F2, lambda F1: b_F2, epsabs=1.0e-3, epsrel=1.0e-3)
    estabilidade = integral/volume
    return estabilidade

def entropia_diferencial(kde, params):
    a_F1 = params['min_F1']
    b_F1 = params['max_F1']
    a_F2 = params['min_F2']
    b_F2 = params['max_F2']
    def integrando(F1, F2):
        fdp = kde([F1, F2])[0]
        epsilon = 1e-10
        return -fdp * np.log(fdp + epsilon)
    entropia, erro = nquad(integrando, [[a_F1, b_F1], [a_F2, b_F2]])
    return entropia

def equacao_maxent(F1, F2, fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params):
    RA = np.exp( restricao_articulatoria(F1, F2, kde, params) - kde([F1, F2])[0] )
    RP = np.exp( restricao_perceptual(F1, kde, params) - fdp_marginalizada(F1) )
    formula_maxent = np.exp(-1 - lambda_zero - lambda_RA * RA - lambda_RP * RP)
    return formula_maxent

def calculo_maxent(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params):
    def fdp_maxent(F1, F2):
        maxent = equacao_maxent(F1, F2, fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params)
        return maxent
    return fdp_maxent

def arquivo_maxent(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params):
    min_F1 = params['min_F1']
    max_F1 = params['max_F1']
    min_F2 = params['min_F2']
    max_F2 = params['max_F2']
    resolucao = params['resolucao']
    fdp_maxent = calculo_maxent(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params)
    valores_F1 = np.linspace(min_F1, max_F1, resolucao)
    valores_F2 = np.linspace(min_F2, max_F2, resolucao)
    grade_F1, grade_F2 = np.meshgrid(valores_F1, valores_F2)
    with open('MaxEnt_values.txt', 'w') as file:
        file.write("F1, F2, Valor MaxEnt\n")
        for i in range(resolucao):
            for j in range(resolucao):
                valor_F1 = grade_F1[i, j]
                valor_F2 = grade_F2[i, j]
                valor_maxent = fdp_maxent(valor_F1, valor_F2)
                file.write(f"{valor_F1}, {valor_F2}, {valor_maxent}\n")

def fdps_otimizacao(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params):
    print('\n---> LAMBDAS <---')
    print('---> Lambda zero:', lambda_zero)
    print('---> Lambda RA:', lambda_RA,)
    print('---> Lambda RP:', lambda_RP)
    print('\n--- START OF OPTIMIZATION ITERATION ---')
    min_F1 = params['min_F1']
    max_F1 = params['max_F1']
    min_F2 = params['min_F2']
    max_F2 = params['max_F2']
    resolucao = params['resolucao']
    fdp_maxent = calculo_maxent(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params)
    print('OPTIMIZATION: MAXENT COMPUTATION COMPLETED')
    grade_F1 = np.linspace(min_F1, max_F1, resolucao)
    grade_F2 = np.linspace(min_F2, max_F2, resolucao)
    grade_F1, grade_F2 = np.meshgrid(grade_F1, grade_F2)
    print('OPTIMIZATION: GRID CREATED')
    pontos_grade = np.vstack([grade_F1.ravel(), grade_F2.ravel()])
    print('OPTIMIZATION: STACKED GRID POINTS INTO A 2D ARRAY')
    valores_kde = kde(pontos_grade).reshape(grade_F1.shape)
    print('OPTIMIZATION: DATA PDF VALUES CREATED')
    valores_maxent = np.zeros_like(grade_F1)
    print('OPTIMIZATION: INITIALIZED 2D ARRAY FOR MAXENT VALUES')
    print('OPTIMIZATION: STARTING TO FILL ARRAY WITH MAXENT VALUES')
    for i in range(grade_F1.shape[0]):
        for j in range(grade_F1.shape[1]):
            valor_F1 = grade_F1[i, j]
            valor_F2 = grade_F2[i, j]
            valores_maxent[i, j] = fdp_maxent(valor_F1, valor_F2)
    print('OPTIMIZATION: MAXENT VALUES CREATED')
    print('--- END OF OPTIMIZATION ITERATION ---')
    return valores_maxent, valores_kde

def criar_fdps(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params):
    min_F1 = params['min_F1']
    max_F1 = params['max_F1']
    min_F2 = params['min_F2']
    max_F2 = params['max_F2']
    resolucao = params['resolucao']
    fdp_maxent = calculo_maxent(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params)
    grade_F1 = np.linspace(min_F1, max_F1, resolucao)
    grade_F2 = np.linspace(min_F2, max_F2, resolucao)
    grade_F1, grade_F2 = np.meshgrid(grade_F1, grade_F2)
    pontos_grade = np.vstack([grade_F1.ravel(), grade_F2.ravel()])
    valores_kde = kde(pontos_grade).reshape(grade_F1.shape)
    valores_maxent = np.zeros_like(grade_F1)
    for i in range(grade_F1.shape[0]):
        for j in range(grade_F1.shape[1]):
            valor_F1 = grade_F1[i, j]
            valor_F2 = grade_F2[i, j]
            valores_maxent[i, j] = fdp_maxent(valor_F1, valor_F2)
    dx = (max_F1 - min_F1) / resolucao
    dy = (max_F2 - min_F2) / resolucao
    integral_maxent = np.sum(valores_maxent) * dx * dy
    print('MaxEnt PDF integral before normalization:', integral_maxent)
    integral_kde = np.sum(valores_kde) * dx * dy
    print('Data PDF integral before normalization:', integral_kde)
    if integral_maxent != 1:
        valores_maxent /= integral_maxent
    if integral_kde != 1:
        valores_kde /= integral_kde
    integral_maxent_posn = np.sum(valores_maxent) * dx * dy
    print('MaxEnt PDF integral after normalization:', integral_maxent_posn)
    integral_kde_posn = np.sum(valores_kde) * dx * dy
    print('Data PDF integral after normalization:', integral_kde_posn)
    return valores_maxent, valores_kde

def verificacao_fdps(valores_maxent, valores_kde_dimensionados, params):
    fator_superior = 1
    fator_inferior = 1
    min_F1 = params['min_F1'] * fator_inferior
    max_F1 = params['max_F1'] * fator_superior
    min_F2 = params['min_F2'] * fator_inferior
    max_F2 = params['max_F2'] * fator_superior
    resolucao = params['resolucao']
    grade_F1 = np.linspace(min_F1, max_F1, resolucao)
    grade_F2 = np.linspace(min_F2, max_F2, resolucao)
    def calcula_integral_e_verifica_normalizacao(valores):
        integral_F1 = trapz(valores, grade_F1, axis=0)
        integral_total = trapz(integral_F1, grade_F2)
        normalizada = np.isclose(integral_total, 1, atol=0.05)
        return integral_total, normalizada
    integral_maxent, normalizada_maxent = calcula_integral_e_verifica_normalizacao(valores_maxent)
    integral_kde, normalizada_kde = calcula_integral_e_verifica_normalizacao(valores_kde_dimensionados)
    print("\nPDF verification and comparison results:")
    print(f"MaxEnt integral: {integral_maxent}, Normalized: {'Yes' if normalizada_maxent else 'No'}")
    print(f"KDE integral: {integral_kde}, Normalized: {'Yes' if normalizada_kde else 'No'}")
    return integral_maxent, integral_kde, normalizada_maxent, normalizada_kde

def kullback_leibler(valores_maxent, valores_kde, params):
    epsilon = 1e-10
    valores_maxent = np.clip(valores_maxent, epsilon, None)
    valores_kde = np.clip(valores_kde, epsilon, None)
    min_F1 = params['min_F1']
    max_F1 = params['max_F1']
    min_F2 = params['min_F2']
    max_F2 = params['max_F2']
    resolucao = params['resolucao']
    grade_F1 = np.linspace(min_F1, max_F1, resolucao)
    grade_F2 = np.linspace(min_F2, max_F2, resolucao)
    dx = np.abs(grade_F1[1] - grade_F1[0])
    dy = np.abs(grade_F2[1] - grade_F2[0])
    def normalizar_fdp(valores, dx, dy):
        if valores.ndim == 2:
            soma_integral = np.trapz(np.trapz(valores, dx=dx, axis=1), dx=dy)
        elif valores.ndim == 1:
            soma_integral = np.trapz(valores, dx=dx)
        else:
            raise ValueError("Erro: o input deve ser 1D ou 2D")
        return valores / soma_integral
    valores_maxent_normalizados = normalizar_fdp(valores_maxent, dx, dy)
    valores_kde_normalizados = normalizar_fdp(valores_kde, dx, dy)
    def verificar_integral_fdp_normalizada(valores, dx, dy, nome_fdp):
        if valores.ndim == 2:
            integral = np.sum(np.trapz(np.trapz(valores, dx=dx, axis=1), dx=dy))
        elif valores.ndim == 1:
            integral = np.trapz(valores, dx=dx)
        print(f"Integral of normalized {nome_fdp}: {integral}")
    verificar_integral_fdp_normalizada(valores_maxent_normalizados, dx, dy, "MaxEnt")
    verificar_integral_fdp_normalizada(valores_kde_normalizados, dx, dy, "EDK")
    p_log_p = valores_kde_normalizados * np.log(valores_kde_normalizados)
    p_log_q = valores_kde_normalizados * np.log(valores_maxent_normalizados)
    def integral_kl(valores, dx, dy):
        integral_intermediaria = np.trapz(valores, dx=dx, axis=1)
        return np.trapz(integral_intermediaria, dx=dy)
    integral_p_log_p = integral_kl(p_log_p, dx, dy)
    integral_p_log_q = integral_kl(p_log_q, dx, dy)
    divergencia_kl = integral_p_log_p - integral_p_log_q
    return divergencia_kl

class OtimizadorKL:
    def __init__(self, kde, params_normalizados_ref):
        self.kde = kde
        self.params = params_normalizados_ref
        self.contador_zero = {'Lambda_zero': 0, 'RA': 0, 'RP': 0}
    def funcao_objetivo(self, lambdas):
        print('Zero counter:', self.contador_zero)
        min_F1 = self.params['min_F1']
        max_F1 = self.params['max_F1']
        min_F2 = self.params['min_F2']
        max_F2 = self.params['max_F2']
        resolucao = self.params['resolucao']
        lambda_zero, peso_RA, peso_RP = lambdas[0], lambdas[1], lambdas[2]
        epsilon = 1e-4
        limiar_zero = 1
        if lambda_zero <= 0:
            self.contador_zero['Lambda_zero'] += 1
            if self.contador_zero['Lambda_zero'] >= limiar_zero:
                lambda_zero = epsilon
        else:
            self.contador_zero['Lambda_zero'] = 0
        if peso_RA <= 0:
            self.contador_zero['RA'] += 1
            if self.contador_zero['RA'] >= limiar_zero:
                peso_RA = epsilon
        else:
            self.contador_zero['RA'] = 0
        if peso_RP <= 0:
            self.contador_zero['RP'] += 1
            if self.contador_zero['RP'] >= limiar_zero:
                peso_RP = epsilon
        else:
            self.contador_zero['RP'] = 0
        valores_maxent, valores_kde = fdps_otimizacao(fdp_marginalizada, lambda_zero, peso_RA, peso_RP, kde, self.params)
        print('\nCalculating KL divergence')
        divergencia_kl = kullback_leibler(valores_maxent, valores_kde, self.params)
        print('\n-----> KL divergence:', divergencia_kl)
        return divergencia_kl

def extrair_probabilidades(valores_maxent, params, scaler):
    print('\n----')
    min_F1, max_F1 = params['min_F1'], params['max_F1']
    min_F2, max_F2 = params['min_F2'], params['max_F2']
    a_F1, a_F2 = params['a_F1'], params['a_F2']
    b_F1, b_F2 = params['b_F1'], params['b_F2']
    resolucao = params['resolucao']
    grade_F1 = np.linspace(min_F1, max_F1, resolucao)
    grade_F2 = np.linspace(min_F2, max_F2, resolucao)
    dx = grade_F1[1] - grade_F1[0]
    dy = grade_F2[1] - grade_F2[0]
    marginal_F1_fdp = np.trapz(valores_maxent, dx=dy, axis=1)
    marginal_F2_fdp = np.trapz(valores_maxent, dx=dx, axis=0)
    marginal_F1_fdp_normalized = marginal_F1_fdp / np.trapz(marginal_F1_fdp, dx=dx)
    marginal_F2_fdp_normalized = marginal_F2_fdp / np.trapz(marginal_F2_fdp, dx=dy)
    indices_F1 = np.logical_and(grade_F1 >= a_F1, grade_F1 <= b_F1)
    indices_F2 = np.logical_and(grade_F2 >= a_F2, grade_F2 <= b_F2)
    relevant_valores_maxent = valores_maxent[np.ix_(indices_F1, indices_F2)]
    prob_conjunta = np.trapz(np.trapz(relevant_valores_maxent, dx=dx, axis=1), dx=dy)
    prob_F1 = np.trapz(marginal_F1_fdp_normalized[indices_F1], dx=dx)
    prob_F2 = np.trapz(marginal_F2_fdp_normalized[indices_F2], dx=dy)
    print(f"Joint probability F1, F2: {prob_conjunta:.9f}")
    print(f"Probability F1: {prob_F1:.12f}")
    print(f"Probability F2: {prob_F2:.12f}")
    print('----')
    return prob_conjunta, prob_F1, prob_F2

params_dados, params_pesos, otimizar, params_arq = parse_parametros()
entrevistados = params_arq['entrevistados']
vogais = params_arq['vogais']
caminho_do_arquivo = params_arq['caminho_do_arquivo']

candidatos = ler_dados(caminho_do_arquivo, vogais, entrevistados)

print('\nFile read successfully:', caminho_do_arquivo)

print('\nSelected speaker(s):', entrevistados)

print('\nSelected vowel(s):', vogais)


largura_customizada = 0.15
largura_scott = 'scott'
largura_silverman = 'silverman'
kde, scaler = calcular_kde(candidatos)

print('\nThe p.d.f. values were estimated successfully using KDE.')

params_dados['resolucao'] = 1000
params_dados['min_F1'] = min(candidatos['F1'])
params_dados['max_F1'] = max(candidatos['F1'])
params_dados['min_F2'] = min(candidatos['F2'])
params_dados['max_F2'] = max(candidatos['F2'])

params_normalizados = {
    'resolucao':params_dados['resolucao'],
    'alvo_F1': scaler.transform([[params_dados['alvo_F1'], 0]])[0][0],
    'limiar_1': scaler.transform([[params_dados['limiar_1'], 0]])[0][0],
    'limiar_2': scaler.transform([[params_dados['limiar_2'], 0]])[0][0],
    'neutro_F1': scaler.transform([[params_dados['neutro_F1'], 0]])[0][0],
    'neutro_F2': scaler.transform([[params_dados['neutro_F2'], 0]])[0][0],
    'a_F1': scaler.transform([[params_dados['a_F1'], 0]])[0][0],
    'b_F1': scaler.transform([[params_dados['b_F1'], 0]])[0][0],
    'min_F1': scaler.transform([[params_dados['min_F1'], 0]])[0][0],
    'max_F1': scaler.transform([[params_dados['max_F1'], 0]])[0][0],
    'alvo_F2': scaler.transform([[0, params_dados['alvo_F2']]])[0][1],
    'a_F2': scaler.transform([[0, params_dados['a_F2']]])[0][1],
    'b_F2': scaler.transform([[0, params_dados['b_F2']]])[0][1],
    'min_F2': scaler.transform([[0, params_dados['min_F2']]])[0][1],
    'max_F2': scaler.transform([[0, params_dados['max_F2']]])[0][1],
    'L': params_dados['L'],
    'k_1': params_dados['k_1'],
    'k_2': params_dados['k_2'],
}
params_ref = {
        'resolucao':params_dados['resolucao'],
        'alvo_F1': params_dados['alvo_F1'],
        'alvo_F2': params_dados['alvo_F2'],
        'limiar_1': params_dados['limiar_1'],
        'limiar_2': params_dados['limiar_2'],
        'neutro_F1': params_dados['neutro_F1'],
        'neutro_F2': params_dados['neutro_F2'],
        'L': params_dados['L'],
        'k_1': params_dados['k_1'],
        'k_2': params_dados['k_2'],
        'a_F1': candidatos['F1'].min(),
        'b_F1': candidatos['F1'].max(),
        'a_F2': candidatos['F2'].min(),
        'b_F2': candidatos['F2'].max(),
        'min_F1': candidatos['F1'].min(),
        'max_F1': candidatos['F1'].max(),
        'min_F2': candidatos['F2'].min(),
        'max_F2': candidatos['F2'].max(),
    }
params_normalizados_ref = {
    'resolucao':1000,
    'alvo_F1': scaler.transform([[params_ref['alvo_F1'], 0]])[0][0],
    'limiar_1': scaler.transform([[params_ref['limiar_1'], 0]])[0][0],
    'limiar_2': scaler.transform([[params_ref['limiar_2'], 0]])[0][0],
    'neutro_F1': scaler.transform([[params_ref['neutro_F1'], 0]])[0][0],
    'neutro_F2': scaler.transform([[params_ref['neutro_F2'], 0]])[0][0],
    'a_F1': scaler.transform([[params_ref['a_F1'], 0]])[0][0],
    'b_F1': scaler.transform([[params_ref['b_F1'], 0]])[0][0],
    'min_F1': scaler.transform([[params_ref['min_F1'], 0]])[0][0],
    'max_F1': scaler.transform([[params_ref['max_F1'], 0]])[0][0],
    'alvo_F2': scaler.transform([[0, params_ref['alvo_F2']]])[0][1],
    'a_F2': scaler.transform([[0, params_ref['a_F2']]])[0][1],
    'b_F2': scaler.transform([[0, params_ref['b_F2']]])[0][1],
    'min_F2': scaler.transform([[0, params_ref['min_F2']]])[0][1],
    'max_F2': scaler.transform([[0, params_ref['max_F2']]])[0][1],
    'L': params_ref['L'],
    'k_1': params_ref['k_1'],
    'k_2': params_ref['k_2'],
    }

arquivo_EDK(kde, params_normalizados_ref)

print('\nFile with a sample of normalized values and the p.d.f. was created.')

fdp_marginalizada = criar_fdp_marginal(kde, params_normalizados_ref)

print('\nMarginalized p.d.f. created.')

entropia = entropia_diferencial(kde, params_normalizados_ref)

SLSQP = 'SLSQP'
BFGS = 'BFGS'
COBYLA = 'COBYLA'
L_BFGS_B = 'L-BFGS-B'

otimizador = OtimizadorKL(kde, params_normalizados_ref)
lambdas_iniciais = [1, 1, 1]
limites = Bounds([1e-8, 1e-8, 1e-8], [np.inf, np.inf, np.inf])

if otimizar:
    print('\nInitializing optimization...')
    otimizacao = minimize(otimizador.funcao_objetivo, lambdas_iniciais, method='L-BFGS-B', bounds=limites, options={'maxiter': 1000})
    lambdas_otimizados = otimizacao.x
    lambda_zero = lambdas_otimizados[0]
    lambda_RA = lambdas_otimizados[1]
    lambda_RP = lambdas_otimizados[2]
    print('\nOptimization completed successfully.')

else:
    lambda_zero = params_pesos['lambda_zero']
    lambda_RA = params_pesos['lambda_RA']
    lambda_RP = params_pesos['lambda_RP']
    print('\nLambda values were specified.')

print('\n----')
print('Constraint weights (lambdas):')
print('0. Weight related to MaxEnt normalization:', lambda_zero)
print('1. Weight of the perceptual constraint:', lambda_RP)
print('2. Weight of the articulatory constraint:', lambda_RA)
print('----')


valor_integral_fdp = integral_fdp(kde, params_normalizados_ref)
violacoes_ra = integral_ra(kde, params_normalizados)
violacoes_ra_pesadas = violacoes_ra*lambda_RA
violacoes_rp = integral_rp(kde, params_normalizados, fdp_marginalizada)
violacoes_rp_pesadas = violacoes_rp*lambda_RP
violacoes_ra_total = integral_ra(kde, params_normalizados_ref)
violacoes_ra_total_pesadas = violacoes_ra_total*lambda_RA
violacoes_rp_total = integral_rp(kde, params_normalizados_ref, fdp_marginalizada)
violacoes_rp_total_pesadas = violacoes_rp_total*lambda_RP
violacoes_ra_normalizado = violacoes_ra_pesadas * 100/violacoes_ra_total_pesadas
violacoes_rp_normalizado = violacoes_rp_pesadas * 100/violacoes_rp_total_pesadas

print('\n----')
print('Candidate:')
print('F1:', params_dados['a_F1'], 'to', params_dados['b_F1'])
print('F2:', params_dados['a_F2'], 'to', params_dados['b_F2'])
print('----')

print('\nCalculating violations...')

print('\nPerceptual constraint violations:', violacoes_rp_pesadas)
print('This represents the share of total possible violations (0-100):', violacoes_rp_normalizado)
print('\nArticulatory constraint violations:', violacoes_ra_pesadas)
print('This represents the share of total possible violations (0-100):', violacoes_ra_normalizado)

print('\nTotal possible Articulatory constraint violations:', violacoes_ra_total_pesadas)
print('Total possible Perceptual constraint violations:', violacoes_rp_total_pesadas)

estabilidade = escore_estabilidade(kde, scaler, fdp_marginalizada, params_normalizados, lambda_RA, lambda_RP)
soma_violacoes = violacoes_ra_pesadas + violacoes_rp_pesadas
soma_violacoes_norm = (violacoes_ra_pesadas + violacoes_rp_pesadas) * 100 / (violacoes_ra_total_pesadas + violacoes_rp_total_pesadas)

print('\nCalculating harmony and stability...')
print('\nStability:', estabilidade)
print('Harmonic score:', soma_violacoes)

print('\nCalculating maximum-entropy probability distribution...')

f_maxent_normalizado = calculo_maxent(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params_normalizados_ref)
fdp_maxent_formatada, fdp_dados_formatada = criar_fdps(fdp_marginalizada, lambda_zero, lambda_RA, lambda_RP, kde, params_normalizados_ref)
verificacao_fdps(fdp_maxent_formatada, fdp_dados_formatada, params_normalizados_ref)

print('\nMaximum-entropy probability distribution created and normalized.')

divergencia_kl = kullback_leibler(fdp_maxent_formatada, fdp_dados_formatada, params_normalizados_ref)

print("\nValue of the Kullback-Leibler divergence:", divergencia_kl)

prob_conjunta, prob_F1, prob_F2 = extrair_probabilidades(fdp_maxent_formatada, params_normalizados, scaler)

def gerar_relatorio(caminho_do_arquivo, vogais, entrevistados, largura_customizada, valor_integral_fdp,
                    violacoes_ra, violacoes_rp, violacoes_ra_total, violacoes_rp_total,
                    violacoes_ra_normalizado, violacoes_rp_normalizado, estabilidade, soma_violacoes,
                    entropia, divergencia_kl, prob_conjunta, prob_F1, prob_F2, a_F1, a_F2, b_F1, b_F2,
                    lambda_zero, lambda_RA, lambda_RP):

    with open('report.txt', 'w', encoding='utf-8') as arquivo:
        arquivo.write("Report\n")
        arquivo.write("------------------------------------------------\n\n")
        arquivo.write(f"Input file: {caminho_do_arquivo}\n\n")
        arquivo.write(f"Vowel(s): {vogais}\n")
        arquivo.write(f"Speaker(s): {entrevistados}\n\n")
        arquivo.write(f'F1: {a_F1} to {b_F1}\n')
        arquivo.write(f'F2: {a_F2} to {b_F2}\n\n')
        arquivo.write(f'Weight of the perceptual constraint (RP): {lambda_RP}\n')
        arquivo.write(f'Weight of the articulatory constraint (RA): {lambda_RA}\n\n')
        arquivo.write(f"RP violations: {violacoes_rp}\n")
        arquivo.write(f"RA violations: {violacoes_ra}\n\n")
        arquivo.write(f"Harmonic score: {soma_violacoes}\n")
        arquivo.write(f"Stability: {estabilidade}\n\n")
        arquivo.write(f"Probability F1: {prob_F1}\n")
        arquivo.write(f"Probability F2: {prob_F2}\n")
        arquivo.write(f"Joint probability of F1 and F2: {prob_conjunta}\n\n")
        arquivo.write(f"KL divergence (model vs. data): {divergencia_kl}\n\n")

gerar_relatorio(caminho_do_arquivo, vogais, entrevistados, largura_customizada, valor_integral_fdp,
                violacoes_ra_pesadas, violacoes_rp_pesadas, violacoes_ra_total, violacoes_rp_total,
                violacoes_ra_normalizado, violacoes_rp_normalizado, estabilidade, soma_violacoes,
                entropia, divergencia_kl, prob_conjunta, prob_F1, prob_F2, params_dados['a_F1'], params_dados['a_F2'],
                    params_dados['b_F1'], params_dados['b_F2'], lambda_zero, lambda_RA, lambda_RP)


print('\nReport file generated and analysis completed.')


def plot_dados_maxent(fdp_maxent, fdp_data, params):


    min_F1 = params['min_F1']
    max_F1 = params['max_F1']
    min_F2 = params['min_F2']
    max_F2 = params['max_F2']
    resolution = params['resolucao']


    x_values = np.linspace(min_F1, max_F1, resolution)
    y_values = np.linspace(min_F2, max_F2, resolution)


    curvatura_F1_maxent = trapz(fdp_maxent, x=x_values, axis=1)
    curvatura_F1_kde = trapz(fdp_data, x=x_values, axis=1)


    integral_F1_maxent = trapz(curvatura_F1_maxent, x=x_values)
    integral_F1_kde = trapz(curvatura_F1_kde, x=x_values)
    curvatura_F1_maxent /= integral_F1_maxent
    curvatura_F1_kde /= integral_F1_kde


    curvatura_F2_maxent = trapz(fdp_maxent, x=y_values, axis=0)
    curvatura_F2_kde = trapz(fdp_data, x=y_values, axis=0)


    integral_F2_maxent = trapz(curvatura_F2_maxent, x=y_values)
    integral_F2_kde = trapz(curvatura_F2_kde, x=y_values)
    curvatura_F2_maxent /= integral_F2_maxent
    curvatura_F2_kde /= integral_F2_kde


    F1_original = scaler.inverse_transform(np.column_stack((x_values, np.zeros_like(x_values))))[:, 0]
    F2_original = scaler.inverse_transform(np.column_stack((np.zeros_like(y_values), y_values)))[:, 1]



    fig1, ax1 = plt.subplots(figsize=(6, 6))


    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


    ax1.plot(F1_original, curvatura_F1_maxent, label='MaxEnt', color='black')
    ax1.plot(F1_original, curvatura_F1_kde, label='Data', color='slateblue')






    ax1.set_xlabel('F1 (Hz)', fontsize=16)
    ax1.set_ylabel('Probability', fontsize=16)


    ax1.legend(fontsize=14)


    plt.savefig('MaxEnt_Data_F1.png')



    fig2, ax2 = plt.subplots(figsize=(6, 6))


    ax2.tick_params(axis='both', which='major', labelsize=16)


    ax2.plot(F2_original, curvatura_F2_maxent, label='MaxEnt', color='black')
    ax2.plot(F2_original, curvatura_F2_kde, label='Data', color='slateblue')


    ax2.set_xlabel('F2 (Hz)', fontsize=16)
    ax2.set_ylabel('Probability', fontsize=16)


    ax2.legend(fontsize=14)


    plt.savefig('MaxEnt_Data_F2.png')


    img_F1 = Image.open('MaxEnt_Data_F1.png')
    img_F2 = Image.open('MaxEnt_Data_F2.png')


    width = img_F1.width + img_F2.width
    height = max(img_F1.height, img_F2.height)


    combined_img = Image.new('RGB', (width, height))


    combined_img.paste(img_F1, (0, 0))
    combined_img.paste(img_F2, (img_F1.width, 0))


    combined_img.save('MaxEnt_Data.png')


    img_F1.close()
    img_F2.close()

plot_dados_maxent(fdp_maxent_formatada, fdp_dados_formatada, params_normalizados_ref)
print('\nGráficos MaxEnt_Data_F1 e MaxEnt_Data_F2 criados.')








def plot_dados_maxent(fdp_maxent, fdp_data, params):


    min_F1 = params['min_F1']
    max_F1 = params['max_F1']
    min_F2 = params['min_F2']
    max_F2 = params['max_F2']
    resolution = params['resolucao']


    x_values = np.linspace(min_F1, max_F1, resolution)
    y_values = np.linspace(min_F2, max_F2, resolution)


    curvatura_F1_maxent = trapz(fdp_maxent, x=x_values, axis=1)
    curvatura_F1_kde = trapz(fdp_data, x=x_values, axis=1)


    integral_F1_maxent = trapz(curvatura_F1_maxent, x=x_values)
    integral_F1_kde = trapz(curvatura_F1_kde, x=x_values)
    curvatura_F1_maxent /= integral_F1_maxent
    curvatura_F1_kde /= integral_F1_kde


    curvatura_F2_maxent = trapz(fdp_maxent, x=y_values, axis=0)
    curvatura_F2_kde = trapz(fdp_data, x=y_values, axis=0)


    integral_F2_maxent = trapz(curvatura_F2_maxent, x=y_values)
    integral_F2_kde = trapz(curvatura_F2_kde, x=y_values)
    curvatura_F2_maxent /= integral_F2_maxent
    curvatura_F2_kde /= integral_F2_kde


    F1_original = scaler.inverse_transform(np.column_stack((x_values, np.zeros_like(x_values))))[:, 0]
    F2_original = scaler.inverse_transform(np.column_stack((np.zeros_like(y_values), y_values)))[:, 1]



    fig1, ax1 = plt.subplots(figsize=(6, 6))


    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


    ax1.plot(F1_original, curvatura_F1_maxent, label='MaxEnt', color='black')



    ax1.set_xlabel('F1 (Hz)', fontsize=16)
    ax1.set_ylabel('Probability', fontsize=16)


    ax1.legend(fontsize=14)


    plt.savefig('MaxEnt_F1.png')



    fig2, ax2 = plt.subplots(figsize=(6, 6))


    ax2.tick_params(axis='both', which='major', labelsize=16)


    ax2.plot(F2_original, curvatura_F2_maxent, label='MaxEnt', color='black')



    ax2.set_xlabel('F2 (Hz)', fontsize=16)
    ax2.set_ylabel('Probability', fontsize=16)


    ax2.legend(fontsize=14)


    plt.savefig('MaxEnt_F2.png')


    img_F1 = Image.open('MaxEnt_F1.png')
    img_F2 = Image.open('MaxEnt_F2.png')


    width = img_F1.width + img_F2.width
    height = max(img_F1.height, img_F2.height)


    combined_img = Image.new('RGB', (width, height))


    combined_img.paste(img_F1, (0, 0))
    combined_img.paste(img_F2, (img_F1.width, 0))


    combined_img.save('MaxEnt.png')


    img_F1.close()
    img_F2.close()

plot_dados_maxent(fdp_maxent_formatada, fdp_dados_formatada, params_normalizados_ref)
print('\nFigures MaxEnt_F1 and MaxEnt_F2 created.')







def plot_divergencia(fdp_maxent, fdp_data, params):


    min_F1 = params['min_F1']
    max_F1 = params['max_F1']
    min_F2 = params['min_F2']
    max_F2 = params['max_F2']
    resolution = params['resolucao']


    x_values = np.linspace(min_F1, max_F1, resolution)
    y_values = np.linspace(min_F2, max_F2, resolution)


    curvatura_F1_maxent = trapz(fdp_maxent, x=x_values, axis=1)
    curvatura_F1_kde = trapz(fdp_data, x=x_values, axis=1)


    integral_F1_maxent = trapz(curvatura_F1_maxent, x=x_values)
    integral_F1_kde = trapz(curvatura_F1_kde, x=x_values)
    curvatura_F1_maxent /= integral_F1_maxent
    curvatura_F1_kde /= integral_F1_kde


    curvatura_F2_maxent = trapz(fdp_maxent, x=y_values, axis=0)
    curvatura_F2_kde = trapz(fdp_data, x=y_values, axis=0)


    integral_F2_maxent = trapz(curvatura_F2_maxent, x=y_values)
    integral_F2_kde = trapz(curvatura_F2_kde, x=y_values)
    curvatura_F2_maxent /= integral_F2_maxent
    curvatura_F2_kde /= integral_F2_kde


    F1_original = scaler.inverse_transform(np.column_stack((x_values, np.zeros_like(x_values))))[:, 0]
    F2_original = scaler.inverse_transform(np.column_stack((np.zeros_like(y_values), y_values)))[:, 1]



    fig1, ax1 = plt.subplots(figsize=(6, 6))


    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


    ax1.plot(F1_original, curvatura_F1_maxent, label='MaxEnt', color='black')
    ax1.plot(F1_original, curvatura_F1_kde, label='Data', color='slateblue')


    ax1.fill_between(F1_original, curvatura_F1_maxent, curvatura_F1_kde, where=(curvatura_F1_maxent > curvatura_F1_kde), color='gray', alpha=0.5, interpolate=True)
    ax1.fill_between(F1_original, curvatura_F1_maxent, curvatura_F1_kde, where=(curvatura_F1_maxent <= curvatura_F1_kde), color='gray', alpha=0.5, interpolate=True)


    ax1.set_xlabel('F1 (Hz)', fontsize=16)
    ax1.set_ylabel('Probability', fontsize=16)


    ax1.legend(fontsize=14)


    plt.savefig('KL_F1.png')



    fig2, ax2 = plt.subplots(figsize=(6, 6))


    ax2.tick_params(axis='both', which='major', labelsize=16)


    ax2.plot(F2_original, curvatura_F2_maxent, label='MaxEnt', color='black')
    ax2.plot(F2_original, curvatura_F2_kde, label='Data', color='slateblue')


    ax2.fill_between(F2_original, curvatura_F2_maxent, curvatura_F2_kde, where=(curvatura_F2_maxent > curvatura_F2_kde), color='gray', alpha=0.5, interpolate=True)
    ax2.fill_between(F2_original, curvatura_F2_maxent, curvatura_F2_kde, where=(curvatura_F2_maxent <= curvatura_F2_kde), color='gray', alpha=0.5, interpolate=True)


    ax2.set_xlabel('F2 (Hz)', fontsize=16)
    ax2.set_ylabel('Probability', fontsize=16)


    ax2.legend(fontsize=14)


    plt.savefig('KL_F2.png')


    img_F1 = Image.open('KL_F1.png')
    img_F2 = Image.open('KL_F2.png')


    width = img_F1.width + img_F2.width
    height = max(img_F1.height, img_F2.height)


    combined_img = Image.new('RGB', (width, height))


    combined_img.paste(img_F1, (0, 0))
    combined_img.paste(img_F2, (img_F1.width, 0))


    combined_img.save('KL.png')


    img_F1.close()
    img_F2.close()

plot_divergencia(fdp_maxent_formatada, fdp_dados_formatada, params_normalizados_ref)
print('\nFigures KL_F1 and KL_F2 created.')







def plot_dados(fdp_data, params, scaler):


    min_F1 = params['min_F1']
    max_F1 = params['max_F1']
    min_F2 = params['min_F2']
    max_F2 = params['max_F2']
    resolution = params['resolucao']


    x_values = np.linspace(min_F1, max_F1, resolution)
    y_values = np.linspace(min_F2, max_F2, resolution)


    curvatura_F1_kde = trapz(fdp_data, x=x_values, axis=1)


    integral_F1_kde = trapz(curvatura_F1_kde, x=x_values)
    curvatura_F1_kde /= integral_F1_kde


    curvatura_F2_kde = trapz(fdp_data, x=y_values, axis=0)


    integral_F2_kde = trapz(curvatura_F2_kde, x=y_values)
    curvatura_F2_kde /= integral_F2_kde


    F1_original = scaler.inverse_transform(np.column_stack((x_values, np.zeros_like(x_values))))[:, 0]
    F2_original = scaler.inverse_transform(np.column_stack((np.zeros_like(y_values), y_values)))[:, 1]



    fig1, ax1 = plt.subplots(figsize=(6, 6))


    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


    ax1.plot(F1_original, curvatura_F1_kde, label='Data', color='black')


    ax1.set_xlabel('F1 (Hz)', fontsize=16)
    ax1.set_ylabel('Density', fontsize=16)


    ax1.legend(fontsize=14)


    plt.savefig('KDE_F1.png')



    fig2, ax2 = plt.subplots(figsize=(6, 6))


    ax2.tick_params(axis='both', which='major', labelsize=16)


    ax2.plot(F2_original, curvatura_F2_kde, label='Data', color='black')


    ax2.set_xlabel('F2 (Hz)', fontsize=16)
    ax2.set_ylabel('Density', fontsize=16)


    ax2.legend(fontsize=14)


    plt.savefig('KDE_F2.png')


    img_F1 = Image.open('KDE_F1.png')
    img_F2 = Image.open('KDE_F2.png')


    width = img_F1.width + img_F2.width
    height = max(img_F1.height, img_F2.height)


    combined_img = Image.new('RGB', (width, height))


    combined_img.paste(img_F1, (0, 0))
    combined_img.paste(img_F2, (img_F1.width, 0))


    combined_img.save('KDE.png')


    img_F1.close()
    img_F2.close()

plot_dados(fdp_dados_formatada, params_normalizados_ref, scaler)
print('\nFigures KDE_F1 and KDE_F2 created.')







def plot_ra(kde, fdp_data, scaler, params):


    pontos_derivada_F1 = [275, 364, 450]
    pontos_derivada_F2 = [1200, 1500, 1800]


    min_F1 = params['min_F1']
    max_F1 = params['max_F1']
    min_F2 = params['min_F2']
    max_F2 = params['max_F2']
    resolucao = params['resolucao']


    valores_F1 = np.linspace(min_F1, max_F1, resolucao)
    valores_F2 = np.linspace(min_F2, max_F2, resolucao)


    curvatura_F1_kde = trapz(fdp_data, x=valores_F1, axis=1)


    integral_F1_kde = trapz(curvatura_F1_kde, x=valores_F1)
    curvatura_F1_kde /= integral_F1_kde


    curvatura_F2_kde = trapz(fdp_data, x=valores_F2, axis=0)


    integral_F2_kde = trapz(curvatura_F2_kde, x=valores_F2)
    curvatura_F2_kde /= integral_F2_kde


    def integrar_F1(F1):
        return quad(lambda F2: restricao_articulatoria(F1, F2, kde, params), min_F2, max_F2)[0]

    def integrar_F2(F2):
        return quad(lambda F1: restricao_articulatoria(F1, F2, kde, params), min_F1, max_F1)[0]


    integral_valores_F1 = [integrar_F1(F1) for F1 in valores_F1]
    integral_valores_F2 = [integrar_F2(F2) for F2 in valores_F2]


    max_F1 = max(np.max(integral_valores_F1), 1)
    max_F2 = max(np.max(integral_valores_F2), 1)
    integral_F1_norm = np.array(integral_valores_F1) / max_F1
    integral_F2_norm = np.array(integral_valores_F2) / max_F2


    F1_orig = scaler.inverse_transform(np.column_stack((valores_F1, np.zeros_like(valores_F1))))[:, 0]
    F2_orig = scaler.inverse_transform(np.column_stack((np.zeros_like(valores_F2), valores_F2)))[:, 1]


    violacoes_F1 = np.exp(integral_F1_norm - curvatura_F1_kde)
    violacoes_F2 = np.exp(integral_F2_norm - curvatura_F2_kde)


    plt.figure(figsize=(6, 6))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(F1_orig, violacoes_F1, label='Violations', color='black')
    plt.plot(F1_orig, integral_F1_norm, label='Constraint', color='black')
    plt.plot(F1_orig, curvatura_F1_kde, label='Data', color='grey')
    plt.fill_between(F1_orig, 0, curvatura_F1_kde, color=(0.9, 0.9, 0.9))
    plt.title('')
    plt.xlabel('F1 (Hz)', size='16')
    plt.ylabel('Output (normalized)', size='16')
    plt.legend(fontsize=14)
    plt.savefig('CA_F1.png')
    plt.close()


    plt.figure(figsize=(6, 6))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(F2_orig, violacoes_F2, label='Violations', color='black')
    plt.plot(F2_orig, integral_F2_norm, label='Constraint', color='black')
    plt.plot(F2_orig, curvatura_F2_kde, label='Data', color='grey')
    plt.fill_between(F2_orig, 0, curvatura_F2_kde, color=(0.9, 0.9, 0.9))
    plt.title('')
    plt.xlabel('F2 (Hz)', size='16')
    plt.ylabel('Output (normalized)', size='16')
    plt.legend(fontsize=14)
    plt.savefig('CA_F2.png')
    plt.close()


    img_F1 = Image.open('CA_F1.png')
    img_F2 = Image.open('CA_F2.png')


    width = img_F1.width + img_F2.width
    height = max(img_F1.height, img_F2.height)


    combined_img = Image.new('RGB', (width, height))


    combined_img.paste(img_F1, (0, 0))
    combined_img.paste(img_F2, (img_F1.width, 0))


    combined_img.save('violations_CA.png')


    img_F1.close()
    img_F2.close()


plot_ra(kde, fdp_dados_formatada, scaler, params_normalizados_ref)
print('\nFigures Derivatives_CA_F2 and Derivatives_CA_F1 created.')







def plot_ra_est(kde, fdp_data, scaler, params):


    pontos_derivada_F1 = [275, 364, 450]
    pontos_derivada_F2 = [1200, 1500, 1800]


    min_F1 = params['min_F1']
    max_F1 = params['max_F1']
    min_F2 = params['min_F2']
    max_F2 = params['max_F2']
    resolucao = params['resolucao']


    valores_F1 = np.linspace(min_F1, max_F1, resolucao)
    valores_F2 = np.linspace(min_F2, max_F2, resolucao)


    curvatura_F1_kde = trapz(fdp_data, x=valores_F1, axis=1)


    integral_F1_kde = trapz(curvatura_F1_kde, x=valores_F1)
    curvatura_F1_kde /= integral_F1_kde


    curvatura_F2_kde = trapz(fdp_data, x=valores_F2, axis=0)


    integral_F2_kde = trapz(curvatura_F2_kde, x=valores_F2)
    curvatura_F2_kde /= integral_F2_kde


    def integrar_F1(F1):
        return quad(lambda F2: restricao_articulatoria(F1, F2, kde, params), min_F2, max_F2)[0]

    def integrar_F2(F2):
        return quad(lambda F1: restricao_articulatoria(F1, F2, kde, params), min_F1, max_F1)[0]


    def integrar_F1_rp(F1):
        return quad(lambda F2: restricao_perceptual(F1, kde, params), min_F1, max_F1)[0]


    integral_valores_F1 = [integrar_F1(F1) + integrar_F1_rp(F1) for F1 in valores_F1]
    integral_valores_F2 = [integrar_F2(F2) for F2 in valores_F2]


    max_F1 = max(np.max(integral_valores_F1), 1)
    max_F2 = max(np.max(integral_valores_F2), 1)
    integral_F1_norm = np.array(integral_valores_F1) / max_F1
    integral_F2_norm = np.array(integral_valores_F2) / max_F2


    F1_orig = scaler.inverse_transform(np.column_stack((valores_F1, np.zeros_like(valores_F1))))[:, 0]
    F2_orig = scaler.inverse_transform(np.column_stack((np.zeros_like(valores_F2), valores_F2)))[:, 1]


    violacoes_F1 = np.exp(integral_F1_norm - curvatura_F1_kde)
    violacoes_F2 = np.exp(integral_F2_norm - curvatura_F2_kde)


    plt.figure(figsize=(6, 6))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(F1_orig, violacoes_F1, label='Harmonic score', color='black')
    plt.axvspan(370, 420, color='grey', alpha=0.3)
    plt.axvspan(490, 540, color='grey', alpha=0.3)

    plt.annotate('', xy=(395, 0.5), xytext=(395, 0.7), arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    plt.annotate('', xy=(515, 0.7), xytext=(515, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    plt.title('')
    plt.xlabel('F1 (Hz)', size='16')
    plt.ylabel('Output (normalized)', size='16')
    plt.legend(fontsize=14)
    plt.savefig('est_F1.png')
    plt.close()


    plt.figure(figsize=(6, 6))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(F2_orig, violacoes_F2, label='Harmonic score', color='black')
    plt.axvspan(1650, 1750, color='grey', alpha=0.3)
    plt.axvspan(1850, 1950, color='grey', alpha=0.3)
    plt.annotate('', xy=(1700, 0.5), xytext=(1700, 0.7), arrowprops=dict(arrowstyle="->", lw=1.5, color="red"))
    plt.annotate('', xy=(1900, 0.7), xytext=(1900, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="red"))
    plt.title('')
    plt.xlabel('F2 (Hz)', size='16')
    plt.ylabel('Output (normalized)', size='16')
    plt.legend(fontsize=14)
    plt.savefig('est_F2.png')
    plt.close()


    img_F1 = Image.open('est_F1.png')
    img_F2 = Image.open('est_F2.png')


    width = img_F1.width + img_F2.width
    height = max(img_F1.height, img_F2.height)


    combined_img = Image.new('RGB', (width, height))


    combined_img.paste(img_F1, (0, 0))
    combined_img.paste(img_F2, (img_F1.width, 0))


    combined_img.save('harmony_CA_est.png')


    img_F1.close()
    img_F2.close()


plot_ra_est(kde, fdp_dados_formatada, scaler, params_normalizados_ref)
print('\nFigure EST_PLOT created.')







def plot_rp(kde, fdp_data, scaler, params):


    pontos_derivada_F1 = [275, 400, 500]


    a_F1 = params['min_F1']
    b_F1 = params['max_F1']
    a_F2 = params['min_F2']
    b_F2 = params['max_F2']
    resolucao = params['resolucao']


    valores_F1 = np.linspace(a_F1, b_F1, resolucao)


    curvatura_F1_kde = trapz(fdp_data, x=valores_F1, axis=1)


    integral_F1_kde = trapz(curvatura_F1_kde, x=valores_F1)
    curvatura_F1_kde /= integral_F1_kde


    def integrar_F1(F1):
        return quad(lambda F2: restricao_perceptual(F1, kde, params), a_F2, b_F2)[0]


    integral_valores_F1 = [integrar_F1(F1) for F1 in valores_F1]


    max_F1 = max(np.max(integral_valores_F1), 1)
    integral_F1_norm = np.array(integral_valores_F1) / max_F1


    F1_orig = scaler.inverse_transform(np.column_stack((valores_F1, np.zeros_like(valores_F1))))[:, 0]


    violacoes = np.exp(integral_F1_norm - curvatura_F1_kde)


    plt.figure(figsize=(12, 6))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(F1_orig, violacoes, label='Violations', color='black')
    plt.plot(F1_orig, integral_F1_norm, label='Constraint', color='black')
    plt.plot(F1_orig, curvatura_F1_kde, label='Data', color='grey')
    plt.fill_between(F1_orig, 0, curvatura_F1_kde, color=(0.9, 0.9, 0.9))
    plt.title('')
    plt.xlabel('F1 (Hz)', size='16')
    plt.ylabel('Output (normalized)', size='16')
    plt.legend(fontsize=14)
    plt.savefig('violations_CP.png')
    plt.close()

plot_rp(kde, fdp_dados_formatada, scaler, params_normalizados_ref)
print('\nFigure CP_F1 created.')







def plot_integrais_ra(valores_kde, kde, scaler, params):


    a_F1, b_F1 = params['min_F1'], params['max_F1']
    a_F2, b_F2 = params['min_F2'], params['max_F2']
    alvo_F2 = params['alvo_F2']
    alvo_F1 = params['alvo_F1']


    valores_F1 = np.linspace(a_F1, b_F1, 1000)
    valores_F2 = np.linspace(a_F2, b_F2, 1000)


    curvatura_F1_kde = np.mean(valores_kde, axis=1)
    curvatura_F2_kde = np.mean(valores_kde, axis=0)


    def integral_F1(F1):

        integrand = lambda F2: kde([F1, F2])[0]
        return  np.exp( restricao_articulatoria(F1, alvo_F2, kde, params) ) / np.exp( quad(integrand, a_F2, b_F2)[0] )


    def integral_F2(F2):

        integrand = lambda F1: kde([F1, F2])[0]
        return  np.exp( restricao_articulatoria(alvo_F1, F2, kde, params) ) / np.exp( quad(integrand, a_F1, b_F1)[0] )



    valores_integral_F1 = np.cumsum([integral_F1(F1) for F1 in valores_F1])
    valores_integral_F2 = np.cumsum([integral_F2(F2) for F2 in valores_F2])


    integral_F1_normalizado = valores_integral_F1 / np.max(valores_integral_F1)
    integral_F2_normalizado = valores_integral_F2 / np.max(valores_integral_F2)


    F1_original = scaler.inverse_transform(np.column_stack((valores_F1, np.zeros_like(valores_F1))))[:, 0]
    F2_original = scaler.inverse_transform(np.column_stack((np.zeros_like(valores_F2), valores_F2)))[:, 1]


    curvatura_F1_kde_normalizado = curvatura_F1_kde / np.max(curvatura_F1_kde) * np.max(integral_F1_normalizado)
    curvatura_F2_kde_normalizado = curvatura_F2_kde / np.max(curvatura_F2_kde) * np.max(integral_F2_normalizado)


    plt.figure(figsize=(6, 6))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(F1_original, integral_F1_normalizado, label='Violations Progression', color='black')
    plt.plot(F1_original, curvatura_F1_kde_normalizado, label='Data', color='slateblue')
    plt.title('')
    plt.xlabel('F1 (Hz)', size='16')
    plt.ylabel('Violations (normalized)', size='16')
    plt.legend(fontsize=14)
    plt.savefig('Integral_CA_F1.png')
    plt.close()


    plt.figure(figsize=(6, 6))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(F2_original, integral_F2_normalizado, label='Violations Progression', color='black')
    plt.plot(F2_original, curvatura_F2_kde_normalizado, label='Data', color='slateblue')
    plt.title('')
    plt.xlabel('F2 (Hz)', size='16')
    plt.ylabel('Violations (normalized)', size='16')
    plt.legend(fontsize=14)
    plt.savefig('Integral_CA_F2.png')
    plt.close()


    img_F1 = Image.open('Integral_CA_F1.png')
    img_F2 = Image.open('Integral_CA_F2.png')


    width = img_F1.width + img_F2.width
    height = max(img_F1.height, img_F2.height)


    combined_img = Image.new('RGB', (width, height))


    combined_img.paste(img_F1, (0, 0))
    combined_img.paste(img_F2, (img_F1.width, 0))


    combined_img.save('Integral_CA.png')


    img_F1.close()
    img_F2.close()

plot_integrais_ra(fdp_dados_formatada, kde, scaler, params_normalizados_ref)
print('\nFigures Integral_CA_F1 and Integral_CA_F2 created successfully.')







def plot_integrais_rp(valores_kde, kde, scaler, params):


    a_F1, b_F1 = params['min_F1'], params['max_F1']
    a_F2, b_F2 = params['min_F2'], params['max_F2']


    valores_F1 = np.linspace(a_F1, b_F1, 1000)


    curvatura_F1_kde = np.mean(valores_kde, axis=1)


    def integral_F1(F1):

        integrand = lambda F2: kde([F1, F2])[0]
        return  np.exp( restricao_perceptual(F1, kde, params) ) / np.exp( quad(integrand, a_F2, b_F2)[0] )


    valores_integral_F1 = np.cumsum([integral_F1(F1) for F1 in valores_F1])


    integral_F1_normalizado = valores_integral_F1 / np.max(valores_integral_F1)


    F1_original = scaler.inverse_transform(np.column_stack((valores_F1, np.zeros_like(valores_F1))))[:, 0]


    curvatura_F1_kde_normalizado = curvatura_F1_kde / np.max(curvatura_F1_kde) * np.max(integral_F1_normalizado)


    plt.figure(figsize=(6, 6))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(F1_original, integral_F1_normalizado, label='Violations Progression', color='black')
    plt.plot(F1_original, curvatura_F1_kde_normalizado, label='Data', color='slateblue')
    plt.title('')
    plt.xlabel('F1 (Hz)', size='16')
    plt.ylabel('Violations (normalized)', size='16')
    plt.legend(fontsize=14)
    plt.savefig('Integral_CP_F1.png')
    plt.close()

plot_integrais_rp(fdp_dados_formatada, kde, scaler, params_normalizados_ref)
print('\nFigure Integral_CA_F1 created successfully.')








def plot_ra_comparativo(valores_kde, kde, scaler, params):


    a_F1, b_F1 = params['min_F1'], params['max_F1']
    a_F2, b_F2 = params['min_F2'], params['max_F2']
    alvo_F2 = params['alvo_F2']
    alvo_F1 = params['alvo_F1']


    valores_F1 = np.linspace(a_F1, b_F1, 1000)


    curvatura_F1_kde = np.mean(valores_kde, axis=1)
    curvatura_F1 = np.mean(valores_kde, axis=1)


    def integral_F1(F1):

        integrand = lambda F2: kde([F1, F2])[0]
        return  np.exp( restricao_articulatoria(F1, alvo_F2, kde, params) ) / np.exp( quad(integrand, a_F2, b_F2)[0] )


    def integral_F1_sem_kde(F1):

        integrand = lambda F2: restricao_articulatoria(F1, F2, kde, params)
        return quad(integrand, a_F2, b_F2)[0]



    valores_integral_F1 = np.cumsum([integral_F1(F1) for F1 in valores_F1])
    valores_integral_F1_sem_kde = np.cumsum([integral_F1_sem_kde(F1) for F1 in valores_F1])


    integral_F1_normalizado = valores_integral_F1 / np.max(valores_integral_F1)
    integral_F1_sem_kde_normalizado = valores_integral_F1_sem_kde / np.max(valores_integral_F1_sem_kde)


    F1_original = scaler.inverse_transform(np.column_stack((valores_F1, np.zeros_like(valores_F1))))[:, 0]


    plt.figure(figsize=(6, 6))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(F1_original, integral_F1_normalizado, label='Violations (with density)', color='black')
    plt.title('')
    plt.xlabel('F1 (Hz)', size='16')
    plt.ylabel('Violations (normalized)', size='16')
    plt.legend(fontsize=14)
    plt.savefig('CA_comparative_1.png')
    plt.close()


    plt.figure(figsize=(6, 6))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(F1_original, integral_F1_sem_kde_normalizado, label='Violations (no density)', color='black')
    plt.title('')
    plt.xlabel('F1 (Hz)', size='16')
    plt.ylabel('Violations (normalized)', size='16')
    plt.legend(fontsize=14)
    plt.savefig('CA_comparative_2.png')
    plt.close()


    img_F1 = Image.open('CA_comparative_1.png')
    img_F2 = Image.open('CA_comparative_2.png')


    width = img_F1.width + img_F2.width
    height = max(img_F1.height, img_F2.height)


    combined_img = Image.new('RGB', (width, height))


    combined_img.paste(img_F1, (0, 0))
    combined_img.paste(img_F2, (img_F1.width, 0))


    combined_img.save('CA_comparative.png')


    img_F1.close()
    img_F2.close()

plot_ra_comparativo(fdp_dados_formatada, kde, scaler, params_normalizados_ref)
print('\nFigures CA_comparative_1 and CA_comparative_2 created successfully.')

