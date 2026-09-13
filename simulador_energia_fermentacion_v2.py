import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Fermentación y refrigeración", page_icon="🍷", layout="wide")
st.title("🍷 Simulador docente de fermentación alcohólica y refrigeración")
st.caption("Modelo cinético–térmico interactivo: azúcar → etanol → calor → necesidades de frío.")

with st.expander("Fundamento y alcance"):
    st.markdown("""
El simulador se inspira en el planteamiento cinético–energético de Palacios,
Udaquiola y Rodríguez (2009), que acopla la evolución de biomasa activa,
nitrógeno, etanol y azúcar con un balance de energía.

Esta implementación es **didáctica y simplificada**: conserva las relaciones
causa–efecto necesarias para experimentar en clase, pero no sustituye un modelo
industrial validado experimentalmente.
""")

SCENARIOS = {
    "Normal (220 g/L; YAN 200)": [10.,220.,200.,0.15,18.,18.,25.,10.],
    "Referencia artículo (291 g/L; N 283)": [10.,291.,283.,0.15,18.,18.,25.,15.],
    "YAN bajo (70 mg/L)": [10.,220.,70.,0.15,18.,18.,25.,10.],
    "Baja temperatura (14 °C)": [10.,220.,200.,0.15,14.,14.,25.,10.],
    "Alta temperatura (25 °C)": [10.,220.,200.,0.15,25.,25.,28.,10.],
    "Refrigeración insuficiente": [20.,250.,220.,0.15,18.,18.,30.,2.5],
}
sc = st.sidebar.selectbox("Escenario docente", list(SCENARIOS))
d = SCENARIOS[sc]

st.sidebar.header("1 · Mosto")
V = st.sidebar.slider("Volumen (m³)",1.,50.,d[0],0.5)
S0 = st.sidebar.slider("Azúcar inicial (g/L)",100.,350.,d[1],5.)
N0 = st.sidebar.slider("YAN / N inicial (mg/L)",20.,400.,d[2],5.)
X0 = st.sidebar.slider("Biomasa activa inicial (g/L)",0.05,0.50,d[3],0.01)

st.sidebar.header("2 · Temperatura")
T0 = st.sidebar.slider("T inicial (°C)",10.,32.,d[4],0.5)
Tset = st.sidebar.slider("T consigna (°C)",10.,30.,d[5],0.5)
Tamb = st.sidebar.slider("T ambiente (°C)",5.,40.,d[6],0.5)

st.sidebar.header("3 · Refrigeración")
Qmax = st.sidebar.slider("Potencia disponible (kW)",0.,50.,d[7],0.5)
control = st.sidebar.toggle("Control de refrigeración", True)
U = st.sidebar.slider("Coeficiente U (W/m²·K)",0.5,12.,3.,0.5)

st.sidebar.header("4 · Tiempo")
hours = st.sidebar.slider("Duración máxima (h)",72,240,168,12)
dt = 0.05

# --- Parámetros docentes ---
mu18 = 0.075
KN = 35.0
KS = 15.0
beta18 = 1.65
Y_E_S = 0.47
Y_X_S = 0.012
N_per_X = 18.0
eth_inhib = 110.0

rho = 1000.0
Cp = 4.0
dH = 0.58
evap_frac = 0.04

# Depósito cilíndrico H/D=2
HD=2.0
D=(4*V/(np.pi*HD))**(1/3)
H=HD*D
A=np.pi*D*H + np.pi*D**2/2
Cth=rho*V*Cp

n=int(hours/dt)+1
t=np.arange(n)*dt
X=np.zeros(n); N=np.zeros(n); E=np.zeros(n); S=np.zeros(n); T=np.zeros(n)
rs=np.zeros(n); Qfer=np.zeros(n); Qamb=np.zeros(n); Qneed=np.zeros(n); Qcool=np.zeros(n)

X[0],N[0],E[0],S[0],T[0]=X0,N0,0.,S0,T0

for i in range(n-1):
    fT=np.clip(2**((T[i]-18)/10),0.30,2.2)
    if T[i] > 30:
        fT *= max(0.05, 1-0.13*(T[i]-30))

    fN=max(N[i],0)/(KN+max(N[i],0))
    fS=max(S[i],0)/(KS+max(S[i],0))
    mu=mu18*fT*fN*fS

    kd=0.001
    if E[i] > eth_inhib:
        kd += 0.00025*(E[i]-eth_inhib)

    growth=mu*X[i]
    death=kd*X[i]

    # Consumo de azúcar: fermentativo + crecimiento.
    sugar_rate=beta18*fT*fS*X[i] + growth/Y_X_S

    # Limitación gradual por N: evita el corte artificial de la versión anterior.
    if N[i] < 40:
        sugar_rate *= 0.55 + 0.45*max(N[i],0)/40

    # Inhibición alcohólica progresiva.
    if E[i] > 115:
        sugar_rate *= max(0.12, 1-0.015*(E[i]-115))

    sugar_rate=min(max(sugar_rate,0),S[i]/dt)
    rs[i]=sugar_rate

    dS=sugar_rate*dt
    dE=Y_E_S*dS
    dX=(growth-death)*dt
    dN=min(N[i],max(growth,0)*N_per_X*dt)

    S[i+1]=max(0,S[i]-dS)
    E[i+1]=E[i]+dE
    X[i+1]=max(0.001,X[i]+dX)
    N[i+1]=max(0,N[i]-dN)

    # Balance térmico
    qfer=sugar_rate*V*1000*dH/3600
    qev=evap_frac*qfer
    qamb=U*A*(Tamb-T[i])/1000

    qbase=max(0,qfer-qev+qamb)
    qrestore=max(0,(T[i]-Tset)*Cth/(2*3600))
    qreq=qbase+qrestore if T[i] >= Tset-0.05 else 0
    qsup=min(Qmax,qreq) if control else 0

    Qfer[i]=qfer; Qamb[i]=qamb; Qneed[i]=qreq; Qcool[i]=qsup
    qnet=qfer-qev+qamb-qsup
    T[i+1]=np.clip(T[i]+qnet*3600*dt/Cth,0,45)

for arr in (rs,Qfer,Qamb,Qneed,Qcool):
    arr[-1]=arr[-2]

rmax=max(rs.max(),1e-9)
phase=[]
for i in range(n):
    if S[i] <= 4:
        phase.append("Finalizada")
    elif rs[i] >= 0.65*rmax:
        phase.append("Máxima actividad")
    elif rs[i] >= 0.25*rmax:
        phase.append("Fermentación activa")
    elif t[i] < 24:
        phase.append("Adaptación / inicio")
    elif S[i] > 20:
        phase.append("Ralentización / posible parada")
    else:
        phase.append("Final de fermentación")
phase=np.array(phase)

energy=float(np.trapezoid(Qcool,t))
peak=float(Qneed.max()); ip=int(np.argmax(Qneed))
abv=E[-1]/7.89
finished=np.where(S<=4)[0]
finish=t[finished[0]] if len(finished) else None
Ncons=N0-N[-1]

st.subheader("Panel de proceso")
view=st.slider("⏱️ Hora visualizada",0.,float(hours),min(72.,float(hours)),1.)
j=int(np.argmin(np.abs(t-view)))

c1,c2,c3,c4=st.columns(4)
c1.metric("Fase",phase[j])
c2.metric("Temperatura",f"{T[j]:.1f} °C",f"Consigna {Tset:.1f} °C")
c3.metric("Azúcar",f"{S[j]:.1f} g/L")
c4.metric("Alcohol",f"{E[j]/7.89:.1f} % vol")

c1,c2,c3,c4=st.columns(4)
c1.metric("YAN",f"{N[j]:.0f} mg/L")
c2.metric("Biomasa",f"{X[j]:.2f} g/L")
c3.metric("Calor fermentativo",f"{Qfer[j]:.2f} kW")
c4.metric("Frío requerido",f"{Qneed[j]:.2f} kW")

if control and Qneed[j] > Qmax:
    st.error("⚠️ La potencia frigorífica disponible es insuficiente en este momento.")
elif control:
    st.success("✓ El equipo frigorífico puede cubrir la demanda actual.")
else:
    st.warning("Refrigeración desactivada.")

st.subheader("Resultados finales")
a,b,c,d=st.columns(4)
a.metric("Azúcar residual",f"{S[-1]:.1f} g/L")
b.metric("Etanol",f"{E[-1]:.1f} g/L")
c.metric("Grado alcohólico",f"{abv:.1f} % vol")
d.metric("Duración",f"{finish:.0f} h" if finish is not None else "No completa")

a,b,c,d=st.columns(4)
a.metric("Pico frigorífico",f"{peak:.2f} kW")
b.metric("Momento del pico",f"{t[ip]:.0f} h")
c.metric("Energía frigorífica",f"{energy:.1f} kWh")
d.metric("N consumido",f"{Ncons:.1f} mg/L",f"{100*Ncons/N0:.1f} %")

tabs=st.tabs(["🍬 Azúcar / etanol","🧫 YAN / biomasa","🌡️ Temperatura","❄️ Refrigeración"])
with tabs[0]:
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=t,y=S,name="Azúcar (g/L)"))
    fig.add_trace(go.Scatter(x=t,y=E,name="Etanol (g/L)"))
    fig.update_layout(xaxis_title="Tiempo (h)",yaxis_title="Concentración (g/L)",height=420)
    st.plotly_chart(fig,use_container_width=True)

with tabs[1]:
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=t,y=N,name="YAN (mg/L)"))
    fig.add_trace(go.Scatter(x=t,y=X,name="Biomasa (g/L)",yaxis="y2"))
    fig.update_layout(xaxis_title="Tiempo (h)",height=420,
                      yaxis=dict(title="YAN (mg/L)"),
                      yaxis2=dict(title="Biomasa (g/L)",overlaying="y",side="right"))
    st.plotly_chart(fig,use_container_width=True)

with tabs[2]:
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=t,y=T,name="Temperatura mosto"))
    fig.add_hline(y=Tset,line_dash="dash",annotation_text="Consigna")
    fig.update_layout(xaxis_title="Tiempo (h)",yaxis_title="°C",height=420)
    st.plotly_chart(fig,use_container_width=True)

with tabs[3]:
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=t,y=Qfer,name="Calor generado"))
    fig.add_trace(go.Scatter(x=t,y=Qneed,name="Frío requerido"))
    fig.add_trace(go.Scatter(x=t,y=Qcool,name="Frío suministrado"))
    fig.add_hline(y=Qmax,line_dash="dash",annotation_text="Potencia disponible")
    fig.update_layout(xaxis_title="Tiempo (h)",yaxis_title="Potencia (kW)",height=420)
    st.plotly_chart(fig,use_container_width=True)

st.subheader("Interpretación docente")
if S[-1] <= 4:
    st.success(f"Fermentación completada. Azúcar residual: {S[-1]:.1f} g/L; alcohol estimado: {abv:.1f} % vol.")
elif S[-1] > 20:
    st.error(f"Fermentación incompleta o lenta. Quedan {S[-1]:.1f} g/L de azúcar.")
else:
    st.warning(f"La fermentación se aproxima al final, pero conserva {S[-1]:.1f} g/L de azúcar.")

if control and peak > Qmax:
    st.error(f"Capacidad frigorífica insuficiente: se requieren hasta {peak:.2f} kW frente a {Qmax:.2f} kW instalados.")

st.subheader("Balance energético")
st.latex(r"\dot Q_{net}=\dot Q_{fer}+\dot Q_{amb}-\dot Q_{evap}-\dot Q_{frio}")
st.latex(r"\frac{dT}{dt}=\frac{\dot Q_{net}}{m C_p}")
st.caption("Pérdida evaporativa adoptada con finalidad docente: 4 % del calor fermentativo.")

df=pd.DataFrame({
    "Tiempo_h":t,"Fase":phase,"Azucar_g_L":S,"Etanol_g_L":E,
    "Alcohol_pct_vol":E/7.89,"YAN_mg_L":N,"Biomasa_g_L":X,
    "Temperatura_C":T,"Consumo_azucar_g_L_h":rs,
    "Calor_fermentacion_kW":Qfer,"Frio_requerido_kW":Qneed,
    "Frio_suministrado_kW":Qcool
})
with st.expander("📋 Datos horarios"):
    step=max(1,int(1/dt))
    st.dataframe(df.iloc[::step].round(3),use_container_width=True,hide_index=True)

st.download_button("⬇️ Descargar resultados CSV",
                   df.to_csv(index=False).encode("utf-8"),
                   "resultados_fermentacion.csv","text/csv")

st.subheader("🎓 Actividad para el alumno")
st.markdown("""
1. Ejecuta el caso normal y localiza el **máximo de actividad fermentativa**.
2. Relaciona la velocidad de consumo de azúcar con el **calor generado**.
3. Compara YAN de 70, 150, 200 y 283 mg/L.
4. Compara consignas de 14, 18 y 25 °C.
5. Duplica el volumen manteniendo las demás condiciones.
6. Reduce la potencia frigorífica hasta que la temperatura deje de seguir la consigna.
7. Explica la diferencia entre **potencia frigorífica (kW)** y **energía frigorífica (kWh)**.
""")

st.warning("Modelo docente simplificado. No utilizar para dimensionamiento industrial sin validación experimental.")
