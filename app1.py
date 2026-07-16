import streamlit as st
import numpy as np
from scipy.optimize import minimize, brentq, minimize_scalar, differential_evolution
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

# ==============================================================================
# 0. SCM VISUALIZATION ENGINE (HTML/JS)
# ==============================================================================
SCM_HTML_ENGINE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SCM Live Simulation</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    
    <style>
        body { 
            font-family: 'Inter', sans-serif; 
            background-color: #0f172a; 
            color: #f8fafc;
            /* Hide body scrollbars for iframe embedding */
            margin: 0;
            overflow: hidden;
            height: 100vh;
        }
        .digital-font { font-family: 'JetBrains Mono', monospace; }
        
        .glass-panel { 
            background: rgba(30, 41, 59, 0.7); 
            border: 1px solid rgba(255,255,255,0.1); 
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }

        /* Custom Range Slider */
        input[type=range] {
            -webkit-appearance: none;
            width: 100%;
            background: transparent;
        }
        input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none;
            height: 16px;
            width: 16px;
            border-radius: 50%;
            background: #38bdf8;
            cursor: pointer;
            margin-top: -6px;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
        }
        input[type=range]::-webkit-slider-runnable-track {
            width: 100%;
            height: 4px;
            cursor: pointer;
            background: #334155;
            border-radius: 2px;
        }

        /* Glowing Progress Rings - Fixed to fit inside boxes */
        .circle-bg {
            fill: none;
            stroke: #1e293b;
            stroke-width: 3;
        }
        .circle {
            fill: none;
            stroke-width: 3;
            stroke-linecap: round;
            transition: stroke-dasharray 0.3s ease-out;
        }
        .circle-blue { stroke: #3b82f6; filter: drop-shadow(0 0 4px #3b82f6); }
        .circle-red { stroke: #ef4444; filter: drop-shadow(0 0 4px #ef4444); }
        .circle-cyan { stroke: #06b6d4; filter: drop-shadow(0 0 4px #06b6d4); }
        .circle-emerald { stroke: #10b981; filter: drop-shadow(0 0 4px #10b981); }
        
        #sim-canvas {
            display: block;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
            border-radius: 1rem;
            box-shadow: inset 0 0 50px rgba(0,0,0,0.5);
        }
    </style>
</head>
<body class="flex flex-col p-4 gap-4 box-border">

    <!-- Header Controls -->
    <header class="flex-shrink-0 grid grid-cols-1 md:grid-cols-6 gap-4 glass-panel rounded-2xl p-4 items-center">
        <div class="col-span-1 md:col-span-2">
            <h1 class="text-xl font-bold text-white tracking-wide">SCM Live Simulation</h1>
            <p class="text-xs text-emerald-400 mt-1 font-semibold">Live Particle & Conditioning Physics</p>
        </div>
        
        <div class="col-span-1 md:col-span-4 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="flex flex-col">
                <div class="flex justify-between">
                    <label class="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Wheat (MT/h)</label>
                    <span id="lbl-grain" class="text-[10px] font-bold text-white digital-font">14.0</span>
                </div>
                <input type="range" id="in-grain" min="2.0" max="20.0" step="0.5" value="14.0" class="mt-1">
            </div>
            <div class="flex flex-col">
                <div class="flex justify-between">
                    <label class="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Speed (RPM)</label>
                    <span id="lbl-rpm" class="text-[10px] font-bold text-amber-400 digital-font">40</span>
                </div>
                <input type="range" id="in-rpm" min="10" max="80" step="1" value="40" class="mt-1">
            </div>
            <div class="flex flex-col">
                <div class="flex justify-between">
                    <label class="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Steam (kg/h)</label>
                    <span id="lbl-steam" class="text-[10px] font-bold text-red-400 digital-font">265</span>
                </div>
                <input type="range" id="in-steam" min="0" max="500" step="5" value="265" class="mt-1">
            </div>
            <div class="flex flex-col">
                <div class="flex justify-between">
                    <label class="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Water Added (L/h)</label>
                    <span id="lbl-water" class="text-[10px] font-bold text-blue-400 digital-font">450</span>
                </div>
                <input type="range" id="in-water" min="0" max="800" step="10" value="450" class="mt-1">
            </div>
        </div>
    </header>

    <!-- Canvas Simulation Area -->
    <main class="flex-1 relative rounded-2xl overflow-hidden border border-slate-700/50 min-h-[300px]">
        <canvas id="sim-canvas"></canvas>
        
        <div class="absolute top-4 right-4 glass-panel px-4 py-2 rounded-lg pointer-events-none text-right">
            <p class="text-[10px] text-slate-400 font-bold uppercase tracking-widest">Active Particles</p>
            <p class="text-2xl digital-font text-sky-400" id="disp-particles">0</p>
        </div>
    </main>

    <!-- FIXED: KPI Footer Rings wrapped in perfectly sized rectangular boxes -->
    <footer class="flex-shrink-0 grid grid-cols-2 md:grid-cols-4 gap-4">
        
        <div class="glass-panel rounded-xl p-3 flex flex-col items-center justify-center relative">
            <div class="relative w-16 h-16 md:w-20 md:h-20 flex items-center justify-center">
                <svg viewBox="0 0 36 36" class="absolute inset-0 w-full h-full">
                    <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                    <path id="ring-base" class="circle circle-blue" stroke-dasharray="100, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                </svg>
                <span id="out-base" class="text-sm md:text-lg font-bold digital-font text-white z-10">10.04</span>
            </div>
            <span class="text-[9px] uppercase font-bold text-blue-400 text-center leading-tight mt-2">Inlet Dry<br>Moisture</span>
        </div>

        <div class="glass-panel rounded-xl p-3 flex flex-col items-center justify-center relative">
            <div class="relative w-16 h-16 md:w-20 md:h-20 flex items-center justify-center">
                <svg viewBox="0 0 36 36" class="absolute inset-0 w-full h-full">
                    <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                    <path id="ring-temp" class="circle circle-red" stroke-dasharray="0, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                </svg>
                <span id="out-temp" class="text-sm md:text-lg font-bold digital-font text-white z-10">--</span>
            </div>
            <span class="text-[9px] uppercase font-bold text-red-400 text-center leading-tight mt-2">SCM Outlet<br>Temp</span>
        </div>

        <div class="glass-panel rounded-xl p-3 flex flex-col items-center justify-center relative">
            <div class="relative w-16 h-16 md:w-20 md:h-20 flex items-center justify-center">
                <svg viewBox="0 0 36 36" class="absolute inset-0 w-full h-full">
                    <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                    <path id="ring-tank" class="circle circle-cyan" stroke-dasharray="0, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                </svg>
                <span id="out-tank" class="text-sm md:text-lg font-bold digital-font text-white z-10">--</span>
            </div>
            <span class="text-[9px] uppercase font-bold text-cyan-400 text-center leading-tight mt-2">Conditioning<br>Moisture</span>
        </div>

        <div class="glass-panel rounded-xl p-3 flex flex-col items-center justify-center relative">
            <div class="relative w-16 h-16 md:w-20 md:h-20 flex items-center justify-center">
                <svg viewBox="0 0 36 36" class="absolute inset-0 w-full h-full">
                    <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                    <path id="ring-atta" class="circle circle-emerald" stroke-dasharray="0, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                </svg>
                <span id="out-atta" class="text-sm md:text-lg font-bold digital-font text-white z-10">--</span>
            </div>
            <span class="text-[9px] uppercase font-bold text-emerald-400 text-center leading-tight mt-2">Final Atta<br>Moisture</span>
        </div>

    </footer>

    <script>
        // ==============================================================================
        // 1. PHYSICS CONSTANTS & ENGINE
        // ==============================================================================
        const R = 8.314;
        const A_s = 0.45;
        const h_fg = 2100000.0;
        const h_coeff = 152.0;
        
        const BASE_MOISTURE = 10.04;
        const FACTORY_EFFICIENCY = 0.93; 
        const MILLING_LOSS = 1.5;

        let LUT = []; 
        let T_env_global = 25.0;
        let T_res_global = 27.0;
        let UI_RESULTS = { outT: 0, tankM: 0, attaM: 0, scmM: 0 };

        function updatePhysicsLUT() {
            const grain_rate = parseFloat(document.getElementById('in-grain').value);
            const water_L_h = parseFloat(document.getElementById('in-water').value);
            const steam_rate = parseFloat(document.getElementById('in-steam').value);
            const rpm = parseFloat(document.getElementById('in-rpm').value);

            const grain_kg_h = grain_rate * 1000.0;

            // 1. SCM Thermodynamic Euler Loop
            T_res_global = 1087.6 / rpm;
            const load_factor = (steam_rate / grain_kg_h) / 0.053;
            T_env_global = steam_rate > 0 ? 30.0 + (116.8 - 30.0) * (1.0 - Math.exp(-1.2 * load_factor)) : 25.0;
            document.getElementById('disp-tenv').innerText = T_env_global.toFixed(1) + '°C';

            LUT = [];
            let T = 25.0;
            let m_w = (BASE_MOISTURE / 100.0) / (1.0 - (BASE_MOISTURE / 100.0));
            const dt = T_res_global / 100.0;

            for(let i=0; i<=100; i++) {
                let M_pct = (m_w / (1.0 + m_w)) * 100.0;
                let Cp = 1300.0 * (1.0 - M_pct/100.0) + 4184.0 * (M_pct/100.0);
                
                let dT = T_env_global - T;
                if (dT > 0) {
                    let q = h_coeff * A_s * dT;
                    T += (q * dt) / Cp;
                    m_w += (q * dt) / h_fg * 0.05; 
                }
                LUT.push({ temp: T, moist: M_pct });
            }

            UI_RESULTS.outT = LUT[100].temp;
            UI_RESULTS.scmM = LUT[100].moist;

            // 2. Downstream Conditioning Tank Mass Balance
            const water_in_wheat = grain_kg_h * (UI_RESULTS.scmM / 100.0);
            const water_absorbed = water_L_h * FACTORY_EFFICIENCY;
            
            const total_water_tank = water_in_wheat + water_absorbed;
            const total_mass_tank = grain_kg_h + water_absorbed; 
            
            const tank_moist_pct = (total_water_tank / total_mass_tank) * 100.0;
            
            UI_RESULTS.tankM = tank_moist_pct;
            UI_RESULTS.attaM = tank_moist_pct - MILLING_LOSS;
            
            updateDashboardRings();
        }

        // ==============================================================================
        // 2. CANVAS ANIMATION ENGINE
        // ==============================================================================
        function tempToColor(t) {
            let ratio = Math.max(0, Math.min((t - 25) / 95, 1.0));
            let r = Math.round(253 - (253 - 120) * ratio);
            let g = Math.round(230 - (230 - 20) * ratio);
            let b = Math.round(138 - (138 - 20) * ratio);
            return `rgb(${r}, ${g}, ${b})`;
        }

        const canvas = document.getElementById('sim-canvas');
        const ctx = canvas.getContext('2d', { alpha: false });
        let CW, CH, M_START, M_END, M_LEN, M_Y, M_HEIGHT, TANK_X, TANK_Y, TANK_W, TANK_H;

        function resize() {
            const rect = canvas.parentElement.getBoundingClientRect();
            if(rect.width === 0) return;
            const dpr = window.devicePixelRatio || 1;
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.setTransform(1, 0, 0, 1, 0, 0); 
            ctx.scale(dpr, dpr);
            
            CW = rect.width; CH = rect.height;

            M_START = Math.max(80, CW * 0.1);
            M_END = Math.min(CW - 150, CW * 0.85);
            M_LEN = M_END - M_START;
            M_Y = CH * 0.35; 
            M_HEIGHT = Math.min(120, CH * 0.35);

            TANK_W = 140;
            TANK_H = CH - (M_Y + M_HEIGHT/2 + 60) - 20;
            TANK_X = M_END - TANK_W/2 + 10;
            TANK_Y = M_Y + M_HEIGHT/2 + 60;
        }
        window.addEventListener('resize', resize);

        class Grain {
            constructor() { this.active = false; }
            reset() {
                this.x = M_START - 20; 
                this.radius = Math.random() * (M_HEIGHT/2 - 10); 
                this.angle = Math.random() * Math.PI * 2;
                this.y = M_Y + Math.sin(this.angle) * this.radius;
                this.vx = 0; this.vy = 0;
                this.size = 3.5 + Math.random() * 1.5;
                this.active = true;
                this.falling = false;
                this.inTank = false;
            }
            update(dt, velocityX, rpm) {
                if(!this.active) return;
                if (!this.falling && !this.inTank) {
                    this.x += velocityX * dt;
                    this.angle += (rpm / 60) * Math.PI * 2 * dt;
                    this.y = M_Y + Math.sin(this.angle) * this.radius;
                    if (this.x >= M_END - 10) {
                        this.falling = true;
                        this.vx = Math.random() * 15; 
                        this.vy = 20 + Math.random() * 50;
                    }
                } else if (this.falling) {
                    this.x += this.vx * dt;
                    this.y += this.vy * dt;
                    this.vy += 600 * dt; 
                    if (this.y > TANK_Y) {
                        this.inTank = true;
                        this.falling = false;
                        this.vy = 10 + Math.random() * 30; 
                    }
                } else if (this.inTank) {
                    this.y += this.vy * dt;
                    this.x += (Math.random() - 0.5) * 10 * dt;
                    if (this.y > TANK_Y + TANK_H - 10) this.active = false; 
                }
            }
            draw() {
                if(!this.active) return;
                let progress = (!this.falling && !this.inTank) ? Math.max(0, Math.min((this.x - M_START) / M_LEN, 1.0)) : 1.0;
                let index = Math.floor(progress * 100);
                index = Math.max(0, Math.min(100, index));
                const state = LUT[index] || LUT[0];
                
                ctx.fillStyle = tempToColor(state.temp);
                ctx.strokeStyle = '#451a03'; 
                ctx.lineWidth = 1;
                ctx.beginPath();
                const depthScale = !this.inTank && !this.falling ? 0.8 + (Math.cos(this.angle) * 0.4) : 1.0;
                ctx.ellipse(this.x, this.y, (this.size * 1.5) * depthScale, this.size * depthScale, this.angle, 0, Math.PI * 2);
                ctx.fill(); ctx.stroke();
            }
        }

        class SteamParticle {
            constructor() { this.active = false; }
            reset() {
                let bias = Math.pow(Math.random(), 2);
                this.x = M_END - (bias * M_LEN);
                this.y = M_Y + (Math.random() - 0.5) * 10;
                this.vx = -(40 + Math.random() * 80); 
                this.vy = (Math.random() - 0.5) * (M_HEIGHT * 1.5); 
                this.life = 1.0;
                this.decay = 0.008 + Math.random() * 0.015;
                this.size = 10 + Math.random() * 20;
                this.active = true;
            }
            update(dt) {
                if(!this.active) return;
                this.x += this.vx * dt; this.y += this.vy * dt;
                this.life -= this.decay; this.size += 30 * dt; 
                if(this.y < M_Y - M_HEIGHT/2) this.y = M_Y - M_HEIGHT/2;
                if(this.y > M_Y + M_HEIGHT/2) this.y = M_Y + M_HEIGHT/2;
                if(this.life <= 0 || this.x < M_START) this.active = false;
            }
            draw(steamRate) {
                if(!this.active) return;
                const intensity = Math.min(1.0, steamRate / 300);
                if (intensity <= 0) return;
                ctx.fillStyle = `rgba(254, 226, 226, ${this.life * 0.4 * intensity})`;
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        class WaterDroplet {
            constructor() { this.active = false; }
            reset() {
                this.x = TANK_X + 20 + (Math.random() * (TANK_W - 40));
                this.y = TANK_Y - 40;
                this.vy = 100 + Math.random() * 100;
                this.active = true;
            }
            update(dt) {
                if(!this.active) return;
                this.y += this.vy * dt;
                if(this.y > TANK_Y + (Math.random() * TANK_H/2)) this.active = false;
            }
            draw(waterRate) {
                if(!this.active || waterRate <= 0) return;
                ctx.fillStyle = `rgba(56, 189, 248, 0.9)`;
                ctx.beginPath();
                ctx.arc(this.x, this.y, 2.5, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        const MAX_GRAINS = 600;
        const grains = Array.from({length: MAX_GRAINS}, () => new Grain());
        const steamParticles = Array.from({length: 200}, () => new SteamParticle());
        const waterParticles = Array.from({length: 80}, () => new WaterDroplet());
        let lastTime = performance.now();

        function renderMachine() {
            ctx.fillStyle = '#1e293b'; ctx.beginPath();
            ctx.roundRect(M_START, M_Y - M_HEIGHT/2, M_LEN, M_HEIGHT, 10); ctx.fill();
            ctx.strokeStyle = '#475569'; ctx.lineWidth = 4; ctx.stroke();

            ctx.fillStyle = '#0f172a'; ctx.fillRect(M_START-20, M_Y - 8, M_LEN+40, 16);
            ctx.fillStyle = '#000';
            for(let i=M_START+20; i<M_END; i+=15) { ctx.beginPath(); ctx.arc(i, M_Y, 2, 0, Math.PI*2); ctx.fill(); }

            ctx.fillStyle = '#1e293b'; ctx.beginPath();
            ctx.roundRect(TANK_X, TANK_Y, TANK_W, TANK_H, 8); ctx.fill(); ctx.stroke();
            ctx.fillStyle = '#0f172a'; ctx.fillRect(TANK_X + 10, TANK_Y + 10, TANK_W - 20, TANK_H - 20);
        }

        function renderPaddles(rpm, time) {
            const numPaddles = Math.floor(M_LEN / 35); 
            const shaftRad = M_HEIGHT / 2 - 10;
            const paddleWidth = 14;
            const rotationAngle = (rpm / 60) * Math.PI * 2 * (time / 1000);

            for(let i=0; i<=numPaddles; i++) {
                const x = M_START + i * 35;
                if(x > M_END - 20) continue;
                const phase = rotationAngle + (i * 0.8);
                const yOffset = Math.sin(phase) * shaftRad;
                const projectedWidth = Math.cos(phase) * paddleWidth;
                
                ctx.strokeStyle = '#334155'; ctx.lineWidth = 4; ctx.beginPath();
                ctx.moveTo(x, M_Y); ctx.lineTo(x, M_Y + yOffset); ctx.stroke();
                ctx.fillStyle = Math.cos(phase) > 0 ? '#94a3b8' : '#475569'; ctx.beginPath();
                ctx.rect(x - Math.abs(projectedWidth)/2, M_Y + yOffset - 4, Math.abs(projectedWidth), 8); ctx.fill();
            }
        }

        function renderInlets() {
            ctx.fillStyle = '#eab308'; ctx.beginPath();
            ctx.moveTo(M_START-30, M_Y - M_HEIGHT/2 - 70); ctx.lineTo(M_START+50, M_Y - M_HEIGHT/2 - 70);
            ctx.lineTo(M_START+30, M_Y - M_HEIGHT/2); ctx.lineTo(M_START-10, M_Y - M_HEIGHT/2); ctx.fill();
            
            ctx.fillStyle = '#ef4444'; ctx.fillRect(M_END + 20, M_Y - 50, 16, 50); ctx.fillRect(M_END + 20, M_Y - 8, 30, 16);  
            
            ctx.fillStyle = '#475569'; ctx.beginPath();
            ctx.moveTo(M_END - 20, M_Y + M_HEIGHT/2); ctx.lineTo(M_END + 20, M_Y + M_HEIGHT/2);
            ctx.lineTo(M_END + 10, TANK_Y); ctx.lineTo(M_END - 30, TANK_Y); ctx.fill();

            ctx.fillStyle = '#38bdf8'; ctx.fillRect(TANK_X + TANK_W/2 - 8, TANK_Y - 60, 16, 60);
            
            ctx.fillStyle = 'rgba(255,255,255,0.08)'; ctx.beginPath();
            ctx.roundRect(M_START, M_Y - M_HEIGHT/2 + 5, M_LEN, 20, 5); ctx.fill();
        }

        function animate(time) {
            const dt = (time - lastTime) / 1000;
            lastTime = time;

            if(CW && CH) { ctx.fillStyle = '#0f172a'; ctx.fillRect(0, 0, CW, CH); }

            const rpm = parseFloat(document.getElementById('in-rpm').value);
            const steamRate = parseFloat(document.getElementById('in-steam').value);
            const waterRate = parseFloat(document.getElementById('in-water').value);
            const grainRate = parseFloat(document.getElementById('in-grain').value);

            renderMachine();
            
            if(steamRate > 0) {
                if(Math.random() < (steamRate/100)) { let dead = steamParticles.find(p => !p.active); if(dead) dead.reset(); }
                steamParticles.forEach(p => { p.update(dt); p.draw(steamRate); });
            }

            renderPaddles(rpm, time);

            if(waterRate > 0) {
                if(Math.random() < (waterRate/150)) { let dead = waterParticles.find(p => !p.active); if(dead) dead.reset(); }
                waterParticles.forEach(p => { p.update(dt); p.draw(waterRate); });
            }

            const targetActiveGrains = Math.min(MAX_GRAINS, (grainRate / 20) * MAX_GRAINS);
            let activeCount = grains.filter(g => g.active).length;
            let toSpawn = Math.min(6, targetActiveGrains - activeCount); 
            if (toSpawn > 0 && rpm > 0 && grainRate > 0) {
                let inactive = grains.filter(g => !g.active);
                for(let i=0; i<toSpawn; i++) if (inactive[i]) inactive[i].reset();
            }

            const velocityX = rpm > 0 ? M_LEN / T_res_global : 0;
            grains.forEach(g => { g.update(dt, velocityX, rpm); g.draw(); });

            document.getElementById('disp-particles').innerText = activeCount;
            renderInlets();

            requestAnimationFrame(animate);
        }

        // ==============================================================================
        // 3. UI UPDATE LOGIC
        // ==============================================================================
        function setRingProgress(id, percentage) {
            const ring = document.getElementById(id);
            if(ring) ring.style.strokeDasharray = `${percentage}, 100`;
        }

        function updateDashboardRings() {
            document.getElementById('out-temp').innerText = UI_RESULTS.outT.toFixed(1);
            setRingProgress('ring-temp', Math.min(100, (UI_RESULTS.outT / 120) * 100));

            document.getElementById('out-tank').innerText = UI_RESULTS.tankM.toFixed(2);
            setRingProgress('ring-tank', Math.max(0, Math.min(100, ((UI_RESULTS.tankM - 9) / 5) * 100)));

            document.getElementById('out-atta').innerText = UI_RESULTS.attaM.toFixed(2);
            setRingProgress('ring-atta', Math.max(0, Math.min(100, ((UI_RESULTS.attaM - 8) / 5) * 100)));
        }

        const inputs = ['in-grain', 'in-water', 'in-steam', 'in-rpm'];
        inputs.forEach(id => {
            document.getElementById(id).addEventListener('input', (e) => {
                document.getElementById(id.replace('in-', 'lbl-')).innerText = parseFloat(e.target.value).toFixed(id==='in-rpm'||id==='in-water'?0:1);
                updatePhysicsLUT();
            });
        });

        resize();
        updatePhysicsLUT();
        requestAnimationFrame(animate);
    </script>
</body>
</html>
"""


# ==============================================================================
# 1. FIXED CONSTANTS & FACTORY DATA
# ==============================================================================
R = 8.314  # Universal Gas Constant (J/mol K)

BASE_PARAMS = {
    "Lipase": {"Eab": 45945.38, "beta": 0.801218, "dH": 117420.10, "dS": 365.4445},
    "Peroxidase": {"Eab": 32662.39, "beta": 0.650912, "dH": 287284.26, "dS": 841.7062},
}

ENZ_TRAIN_DATA = [
    {"temp": 25.00, "moist": 9.40,  "Lipase": 0,      "Peroxidase": 0},
    {"temp": 27.23, "moist": 9.42,  "Lipase": -2.4,  "Peroxidase": -0.34},
    {"temp": 39.27, "moist": 9.63,  "Lipase": 5.05,  "Peroxidase": -0.13},
    {"temp": 52.48, "moist": 9.82,  "Lipase": 7.72,  "Peroxidase": 4.61},
    {"temp": 63.91, "moist": 9.89,  "Lipase": 23.15, "Peroxidase": 36.58},
    {"temp": 73.88, "moist": 10.10, "Lipase": 33.06, "Peroxidase": 57.59},
    {"temp": 77.25, "moist": 10.62, "Lipase": 35.22, "Peroxidase": 48.31},
    {"temp": 76.00, "moist": 10.30, "Lipase": 59.44, "Peroxidase": 93.98},
    {"temp": 82.32, "moist": 11.14, "Lipase": 74.32, "Peroxidase": 98.89},
    {"temp": 84.07, "moist": 11.25, "Lipase": 73.56, "Peroxidase": 96.13},
    {"temp": 83.73, "moist": 10.92, "Lipase": 84.56, "Peroxidase": 97.64}
]

# Updated 12-point dataset containing the lower temperature control benchmarks
GLUT_TEST_TEMPS = np.array([68.0, 68.0, 62.0, 68.0, 65.0, 72.0, 69.0, 66.0, 59.0, 70.0, 38.0, 25.0])
GLUT_TEST_GLUTENS = np.array([8.66, 4.50, 8.65, 7.86, 7.48, 3.35, 7.27, 8.77, 7.46, 8.30, 9.37, 9.08])

# ==============================================================================
# 2. CALIBRATION ENGINES (Cached for performance)
# ==============================================================================
@st.cache_data
def calibrate_enzymes():
    def mse_enz(params, enzyme):
        ln_A0, gamma = params
        A0, base = np.exp(ln_A0), BASE_PARAMS[enzyme]
        error = 0.0
        for pt in ENZ_TRAIN_DATA:
            Tk, M_pct = pt["temp"] + 273.15, pt["moist"]
            Ea_eff = max(base["Eab"] - gamma * M_pct, base["beta"] * base["Eab"])
            k = A0 * np.exp(-Ea_eff / (R * Tk))
            w = 1.0 / (1.0 + np.exp(base["dH"] / (R * Tk) - base["dS"] / R))
            pred = (1.0 - np.exp(-k * w)) * 100.0
            error += (pred - pt[enzyme]) ** 2
        return error / len(ENZ_TRAIN_DATA)

    opt_params = {}
    for enz in ["Lipase", "Peroxidase"]:
        res = minimize(mse_enz, [10.0, 50.0], args=(enz,), method='L-BFGS-B', bounds=[(1.0, 40.0), (0.1, 800.0)])
        opt_params[enz] = {"A0": np.exp(res.x[0]), "gamma": res.x[1]}
    return opt_params

@st.cache_data
def calibrate_gluten():
    # Global Differential Evolution optimizer with relaxed bounds
    def mse_glut(params):
        w_res, w_drop1, w_drop2, t1, t2, k1, k2 = params
        preds = w_res + (w_drop1 / (1.0 + np.exp(k1 * (GLUT_TEST_TEMPS - t1)))) + (w_drop2 / (1.0 + np.exp(k2 * (GLUT_TEST_TEMPS - t2))))
        return np.sqrt(np.mean((preds - GLUT_TEST_GLUTENS)**2))

    bounds = [(1.0, 4.5), (0.1, 7.0), (0.1, 7.0), (50.0, 85.0), (50.0, 85.0), (0.05, 2.0), (0.05, 2.0)]
    res = differential_evolution(mse_glut, bounds=bounds, strategy='best1bin', popsize=20, seed=42)
    return res.x

def calibrate_system(wheat_grain_kg_h, wheat_moist_pct, wheat_after_temp_pct, water_added_L_h, atta_final_moist_pct):
    wheat_moist_frac = wheat_moist_pct / 100.0
    wheat_after_temp_frac = wheat_after_temp_pct / 100.0
    atta_final_moist_frac = atta_final_moist_pct / 100.0
    
    water_initial = wheat_grain_kg_h * wheat_moist_frac
    water_after_tempering_control = (wheat_grain_kg_h + water_added_L_h) * wheat_after_temp_frac
    
    efficiency = (water_after_tempering_control - water_initial) / water_added_L_h
    moisture_loss_milling = (wheat_after_temp_frac - atta_final_moist_frac) * 100.0
    
    return efficiency, moisture_loss_milling

# ==============================================================================
# 3. PREDICTION ENGINES
# ==============================================================================
def predict_enzyme(temp, moist, enzyme, enz_params):
    base = BASE_PARAMS[enzyme]
    Tk = temp + 273.15
    Ea_eff = max(base["Eab"] - enz_params[enzyme]["gamma"] * moist, base["beta"] * base["Eab"])
    k = enz_params[enzyme]["A0"] * np.exp(-Ea_eff / (R * Tk))
    w = 1.0 / (1.0 + np.exp(base["dH"] / (R * Tk) - base["dS"] / R))
    return (1.0 - np.exp(-k * w)) * 100.0

def predict_gluten(temp, glut_params):
    w_res, w_drop1, w_drop2, t1, t2, k1, k2 = glut_params
    return w_res + (w_drop1 / (1.0 + np.exp(k1 * (temp - t1)))) + (w_drop2 / (1.0 + np.exp(k2 * (temp - t2))))

def get_optimal_window(target_lipase, target_perox, target_gluten, moisture, enz_params, glut_params):
    try: t_lip = brentq(lambda t: predict_enzyme(t, moisture, "Lipase", enz_params) - target_lipase, 10, 150)
    except ValueError: t_lip = None
    try: t_pox = brentq(lambda t: predict_enzyme(t, moisture, "Peroxidase", enz_params) - target_perox, 10, 150)
    except ValueError: t_pox = None
    try: t_glut = brentq(lambda t: predict_gluten(t, glut_params) - target_gluten, 20, 120)
    except ValueError: t_glut = None
    return t_lip, t_pox, t_glut

# ==============================================================================
# 4. THERMODYNAMIC SOLVERS
# ==============================================================================
def simulate_steamer(steam_rate, water_actual_L_h, efficiency, grain_rate_mt_h, rpm, inlet_temp, inlet_moist_pct):
    grain_kg_h = grain_rate_mt_h * 1000.0
    inlet_moist_frac = inlet_moist_pct / 100.0
    
    water_initial = grain_kg_h * inlet_moist_frac
    water_effectively_absorbed = water_actual_L_h * efficiency
    
    total_water_after_temp = water_initial + water_effectively_absorbed
    total_mass_after_temp = grain_kg_h + water_actual_L_h
    tempered_moisture_pct = (total_water_after_temp / total_mass_after_temp) * 100.0
    
    t_res = 1087.6 / rpm
    psi = steam_rate / grain_kg_h if grain_kg_h > 0 else 0
    load_factor = psi / 0.053
    T_env = 30.0 + (116.8 - 30.0) * (1.0 - np.exp(-1.2 * load_factor))
    
    T = inlet_temp
    dt = 0.1
    steps = int(t_res / dt)
    
    m_w = (tempered_moisture_pct / 100.0) / (1.0 - (tempered_moisture_pct / 100.0))
    m_w_start = m_w
    
    for _ in range(steps):
        M_pct_now = (m_w / (1.0 + m_w)) * 100.0
        Cp_dynamic = 1300.0 * (1.0 - M_pct_now/100.0) + 4184.0 * (M_pct_now/100.0)
        
        dT = T_env - T
        if dT > 0:
            q = 152.0 * 0.45 * dT 
            T += (q * dt) / Cp_dynamic
            m_w += (q * dt) / 2100000.0 * 0.05 
            
    dry_weight = grain_kg_h * (1.0 - inlet_moist_frac)
    water_added_by_steam_kg_h = (m_w - m_w_start) * dry_weight
    return T, water_added_by_steam_kg_h

def get_required_inputs_with_steam_moisture(target_temp, target_moist, moist_type, rpm, grain_rate, efficiency, moisture_loss, inlet_temp, inlet_moist_pct):
    grain_kg_h = grain_rate * 1000.0
    inlet_frac = inlet_moist_pct / 100.0
    
    if moist_type.lower() == 'atta':
        moisture_after_temp_pct = target_moist + moisture_loss
    else:
        moisture_after_temp_pct = target_moist
        
    moist_after_temp_frac = moisture_after_temp_pct / 100.0
    
    dry_weight = grain_kg_h * (1.0 - inlet_frac)
    water_before_tempering = grain_kg_h * inlet_frac
    water_after_tempering = dry_weight * moist_after_temp_frac / (1.0 - moist_after_temp_frac)
    total_water_for_tempering = water_after_tempering - water_before_tempering
    
    W_steam = 0.0
    opt_water = 0.0
    opt_steam = 0.0
    
    for _ in range(5):
        liquid_water_needed = max(0, total_water_for_tempering - W_steam)
        water_actual = liquid_water_needed / efficiency
        
        def objective(steam_guess):
            T_sim, _ = simulate_steamer(steam_guess, water_actual, efficiency, grain_rate, rpm, inlet_temp, inlet_moist_pct)
            return (T_sim - target_temp)**2
            
        res = minimize_scalar(objective, bounds=(0, 1500), method='bounded')
        opt_steam = res.x
        
        _, W_steam_new = simulate_steamer(opt_steam, water_actual, efficiency, grain_rate, rpm, inlet_temp, inlet_moist_pct)
        
        if abs(W_steam_new - W_steam) < 0.01:
            W_steam = W_steam_new
            opt_water = water_actual
            break
            
        W_steam = W_steam_new

    return opt_steam, opt_water, W_steam

def predict_outcomes_from_inputs(steam_rate_kg_h, water_added_L_h, grain_rate_mt_h, rpm, inlet_temp, inlet_moist_pct, efficiency, moisture_loss_milling, enz_params, glut_params):
    # 1. Thermodynamics
    final_temp, water_from_steam_kg_h = simulate_steamer(
        steam_rate_kg_h, water_added_L_h, efficiency, grain_rate_mt_h, rpm, inlet_temp, inlet_moist_pct
    )

    # 2. Corrected Mass Balance
    grain_flow_kg_h = grain_rate_mt_h * 1000.0
    inlet_moist_frac = inlet_moist_pct / 100.0
    
    water_initial = grain_flow_kg_h * inlet_moist_frac
    water_effectively_absorbed = water_added_L_h * efficiency
    
    total_water_after_temp = water_initial + water_effectively_absorbed + water_from_steam_kg_h
    total_mass_after_temp = grain_flow_kg_h + water_added_L_h + water_from_steam_kg_h
    
    tempered_moisture_frac = total_water_after_temp / total_mass_after_temp
    tempered_moisture_pct = tempered_moisture_frac * 100.0
    
    final_atta_moisture_pct = tempered_moisture_pct - moisture_loss_milling

    # 3. Biology
    lipase_inact = predict_enzyme(final_temp, tempered_moisture_pct, "Lipase", enz_params)
    perox_inact = predict_enzyme(final_temp, tempered_moisture_pct, "Peroxidase", enz_params)
    gluten_retention = predict_gluten(final_temp, glut_params)

    return {
        "final_temperature_c": final_temp,
        "tempered_wheat_moisture_pct": tempered_moisture_pct,
        "final_atta_moisture_pct": final_atta_moisture_pct,
        "lipase_inactivated_pct": lipase_inact,
        "peroxidase_inactivated_pct": perox_inact,
        "gluten_retained_pct": gluten_retention,
        "steam_condensed_L_h": water_from_steam_kg_h
    }

# ==============================================================================
# 5. VISUALIZATION ENGINES
# ==============================================================================
def generate_plotly_fig(moisture, t_lip, t_pox, t_glut, target_l, target_p, target_g, enz_params, glut_params):
    max_t = max([100, t_lip or 0, t_pox or 0, t_glut or 0]) + 10
    temps_array = np.linspace(20, max_t, 300)
    temps = temps_array.tolist() 
    
    lip_vals = [predict_enzyme(t, moisture, "Lipase", enz_params) for t in temps]
    pox_vals = [predict_enzyme(t, moisture, "Peroxidase", enz_params) for t in temps]
    glut_vals = [predict_gluten(t, glut_params) for t in temps]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=temps, y=lip_vals, mode='lines', name='Lipase Inact.', line=dict(color='blue', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=temps, y=pox_vals, mode='lines', name='Peroxidase Inact.', line=dict(color='orange', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=temps, y=glut_vals, mode='lines', name='Gluten Retention', line=dict(color='purple', width=3, dash='dash')), secondary_y=True)

    min_required_temp = max(t_lip or 0, t_pox or 0) if (t_lip or t_pox) else None
    max_allowed_temp = t_glut

    if min_required_temp and max_allowed_temp:
        if min_required_temp <= max_allowed_temp:
            fig.add_vrect(x0=min_required_temp, x1=max_allowed_temp, fillcolor="green", opacity=0.15, layer="below", line_width=0, annotation_text="Valid Window")
        else:
            fig.add_vrect(x0=max_allowed_temp, x1=min_required_temp, fillcolor="red", opacity=0.15, layer="below", line_width=0, annotation_text="Conflict Zone")

    fig.add_hline(y=target_l, line_dash="dot", line_color="blue", opacity=0.4, secondary_y=False)
    fig.add_hline(y=target_p, line_dash="dot", line_color="orange", opacity=0.4, secondary_y=False)
    fig.add_hline(y=target_g, line_dash="dot", line_color="purple", opacity=0.4, secondary_y=True)

    if t_lip: fig.add_trace(go.Scatter(x=[t_lip], y=[target_l], mode='markers', name=f'Lipase Target ({t_lip:.1f}°C)', marker=dict(color='blue', size=12, symbol='x')), secondary_y=False)
    if t_pox: fig.add_trace(go.Scatter(x=[t_pox], y=[target_p], mode='markers', name=f'Perox Target ({t_pox:.1f}°C)', marker=dict(color='orange', size=12, symbol='x')), secondary_y=False)
    if t_glut: fig.add_trace(go.Scatter(x=[t_glut], y=[target_g], mode='markers', name=f'Gluten Limit ({t_glut:.1f}°C)', marker=dict(color='purple', size=12, symbol='x')), secondary_y=True)

    fig.update_layout(
        title=f"<b>Optimization Constraints</b><br><sup>Evaluating biological constraints at {moisture}% Target Moisture</sup>",
        xaxis_title="Temperature (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40, l=40, r=40),
        hovermode="x unified",
        height=550
    )
    
    fig.update_yaxes(title_text="Enzyme Inactivation (%)", range=[-5, max(105, target_l + 5, target_p + 5)], secondary_y=False)
    fig.update_yaxes(title_text="Dry Gluten Retention (%)", range=[0, max(12, target_g + 2)], secondary_y=True)
    return fig


def generate_simulation_fig(final_temp, moisture, lipase_val, perox_val, gluten_val, enz_params, glut_params):
    max_t = max(100.0, final_temp + 15.0)
    temps_array = np.linspace(20, max_t, 300)
    temps = temps_array.tolist()
    
    lip_vals = [predict_enzyme(t, moisture, "Lipase", enz_params) for t in temps]
    pox_vals = [predict_enzyme(t, moisture, "Peroxidase", enz_params) for t in temps]
    glut_vals = [predict_gluten(t, glut_params) for t in temps]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=temps, y=lip_vals, mode='lines', name='Lipase Inact.', line=dict(color='blue', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=temps, y=pox_vals, mode='lines', name='Peroxidase Inact.', line=dict(color='orange', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=temps, y=glut_vals, mode='lines', name='Gluten Retention', line=dict(color='purple', width=3, dash='dash')), secondary_y=True)

    # Add a bold vertical line marking exactly where the simulation landed
    fig.add_vline(x=final_temp, line_width=2, line_dash="solid", line_color="red", annotation_text=f"Simulated Temp ({final_temp:.1f}°C)")

    # Add markers at the exact predicted values
    fig.add_trace(go.Scatter(x=[final_temp], y=[lipase_val], mode='markers', name=f'Lipase ({lipase_val:.1f}%)', marker=dict(color='blue', size=12, symbol='x')), secondary_y=False)
    fig.add_trace(go.Scatter(x=[final_temp], y=[perox_val], mode='markers', name=f'Perox ({perox_val:.1f}%)', marker=dict(color='orange', size=12, symbol='x')), secondary_y=False)
    fig.add_trace(go.Scatter(x=[final_temp], y=[gluten_val], mode='markers', name=f'Gluten ({gluten_val:.2f}%)', marker=dict(color='purple', size=12, symbol='x')), secondary_y=True)

    fig.update_layout(
        title=f"<b>Simulation Biological Outcomes</b><br><sup>Displaying enzyme decay and gluten retention at {moisture:.2f}% Tempered Moisture</sup>",
        xaxis_title="Temperature (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40, l=40, r=40),
        hovermode="x unified",
        height=550
    )
    
    fig.update_yaxes(title_text="Enzyme Inactivation (%)", range=[-5, 105], secondary_y=False)
    fig.update_yaxes(title_text="Dry Gluten Retention (%)", range=[0, 12], secondary_y=True)
    return fig

# ==============================================================================
# 6. STREAMLIT UI
# ==============================================================================
st.set_page_config(page_title="Integrated Plant Optimizer", layout="wide")

st.title("🏭 Steam Conveyor Machine Optimizer")
st.markdown("Calculate process minimums to achieve biological targets or predict what will happen with specific machine settings.")

# --- SIDEBAR: CALIBRATION ---
st.sidebar.header("⚙️ Mechanical Calibration")
st.sidebar.markdown("Baseline factory trial data used to calculate tempering efficiency and milling loss.")
cal_grain_flow = st.sidebar.number_input("Grain Flow (kg/h)", value=14000.0, step=100.0)
cal_inlet_moist = st.sidebar.number_input("Calibration Inlet Moisture (%)", value=10.04, step=0.1)
cal_water_added = st.sidebar.number_input("Calibration Water Added (L/h)", value=450.0, step=10.0)
cal_wheat_after_temp = st.sidebar.number_input("Wheat Moisture After Temp (%)", value=11.67, step=0.1)
cal_atta_final = st.sidebar.number_input("Final Atta Moisture (%)", value=9.00, step=0.1)

sys_eff, sys_loss = calibrate_system(cal_grain_flow, cal_inlet_moist, cal_wheat_after_temp, cal_water_added, cal_atta_final)

st.sidebar.markdown("---")
st.sidebar.info(f"**Tempering Efficiency:** {sys_eff*100:.2f}%\n\n**Milling Moisture Loss:** {sys_loss:.2f}%")

enz_params = calibrate_enzymes()
glut_params = calibrate_gluten()

# --- TABS FOR MODES ---
tab1, tab2, tab3 = st.tabs(["🎯 Optimization Mode", "🔮 Simulator Mode", "🎥 SCM Visualization"])

# ------------------------------------------------------------------------------
# TAB 1: OPTIMIZATION MODE
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("⚙️ Current Operating Conditions")
    colA, colB, colC, colD = st.columns(4)
    curr_grain_rate_opt = colA.number_input("Grain Rate (MT/h)", value=5.0, step=0.5, key="opt_grain")
    curr_rpm_opt = colB.number_input("Conveyor RPM", value=40.0, step=1.0, key="opt_rpm")
    curr_inlet_temp_opt = colC.number_input("Inlet Grain Temp (°C)", value=38.0, step=1.0, key="opt_temp")
    curr_inlet_moist_opt = colD.number_input("Inlet Moisture (%)", value=10.04, step=0.1, key="opt_inlet_moist")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧪 Biological Targets")
        tgt_lipase = st.number_input("Desired Lipase Inactivation (%)", value=25.0, step=1.0, key="opt_lipase")
        tgt_perox = st.number_input("Desired Peroxidase Inactivation (%)", value=25.0, step=1.0, key="opt_perox")
        tgt_gluten = st.number_input("Minimum Dry Gluten Retention (%)", value=7.50, step=0.1, key="opt_gluten")
    with col2:
        st.subheader("💧 Moisture Targets")
        tgt_moisture = st.number_input("Target Final Atta Moisture (%)", value=10.7, step=0.1, key="opt_moist")

    if st.button("🚀 Optimize Process & Calculate Mechanical Inputs", use_container_width=True):
        with st.spinner("Simulating Biological Constraints & Solving Thermodynamics..."):
            
            t_lip, t_pox, t_glut = get_optimal_window(tgt_lipase, tgt_perox, tgt_gluten, tgt_moisture, enz_params, glut_params)
            
            fig = generate_plotly_fig(tgt_moisture, t_lip, t_pox, t_glut, tgt_lipase, tgt_perox, tgt_gluten, enz_params, glut_params)
            st.plotly_chart(fig, use_container_width=True)

            if t_lip is None or t_pox is None:
                st.error("❌ Enzyme targets cannot be reached within realistic physical bounds (10-150°C).")
            elif t_glut is None:
                st.error("❌ Gluten target cannot be matched.")
            else:
                min_required_temp = max(t_lip, t_pox)
                max_allowed_temp = t_glut
                
                if min_required_temp <= max_allowed_temp:
                    st.success(f"✅ **Valid Operational Window Found:** Target temperatures range from **{min_required_temp:.1f}°C** to **{max_allowed_temp:.1f}°C**.")
                    status = "Valid"
                else:
                    st.error(f"⚠️ **Conflict Zone Detected:** Minimum temp to kill enzymes is **{min_required_temp:.1f}°C**, but gluten degrades entirely by **{max_allowed_temp:.1f}°C**.")
                    status = "Conflict"

                steam_min_temp, water_min_temp, cond_min = get_required_inputs_with_steam_moisture(
                    target_temp=min_required_temp, target_moist=tgt_moisture, moist_type='Atta', 
                    rpm=curr_rpm_opt, grain_rate=curr_grain_rate_opt, efficiency=sys_eff, 
                    moisture_loss=sys_loss, inlet_temp=curr_inlet_temp_opt, inlet_moist_pct=curr_inlet_moist_opt
                )

                steam_max_temp, water_max_temp, cond_max = get_required_inputs_with_steam_moisture(
                    target_temp=max_allowed_temp, target_moist=tgt_moisture, moist_type='Atta', 
                    rpm=curr_rpm_opt, grain_rate=curr_grain_rate_opt, efficiency=sys_eff, 
                    moisture_loss=sys_loss, inlet_temp=curr_inlet_temp_opt, inlet_moist_pct=curr_inlet_moist_opt
                )

                st.subheader("⚙️ Recommended Mechanical Operating Ranges")
                if status == "Valid":
                    st.caption(f"Operate within these mechanical limits to maintain the temperature safely between **{min_required_temp:.1f}°C** (Lower Limit) and **{max_allowed_temp:.1f}°C** (Upper Limit).")
                    
                    actual_water_low = min(water_min_temp, water_max_temp)
                    actual_water_high = max(water_min_temp, water_max_temp)
                    
                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        st.metric(label="Required Steam Flow Range", value=f"{steam_min_temp:.1f} - {steam_max_temp:.1f} kg/h")
                    with res_col2:
                        st.metric(label="Tempering Water Range", value=f"{actual_water_low:.1f} - {actual_water_high:.1f} L/h")
                    with res_col3:
                        st.metric(label="Water Added via Condensation", value=f"{cond_min:.1f} - {cond_max:.1f} L/h")
                else:
                    st.warning("Because your biological targets conflict, we are showing the mechanical inputs required for the two conflicting extremes.")
                    
                    st.markdown(f"**🔴 Option A: Save the Enzymes (Hits {min_required_temp:.1f}°C, but destroys gluten):**")
                    col_a1, col_a2, col_a3 = st.columns(3)
                    col_a1.metric("Required Steam", f"{steam_min_temp:.1f} kg/h")
                    col_a2.metric("Required Water", f"{water_min_temp:.1f} L/h")
                    col_a3.metric("Condensation Added", f"{cond_min:.2f} L/h")
                    
                    st.markdown("---")
                    
                    st.markdown(f"**🔵 Option B: Save the Gluten (Hits {max_allowed_temp:.1f}°C, but fails to kill enzymes):**")
                    col_b1, col_b2, col_b3 = st.columns(3)
                    col_b1.metric("Required Steam", f"{steam_max_temp:.1f} kg/h")
                    col_b2.metric("Required Water", f"{water_max_temp:.1f} L/h")
                    col_b3.metric("Condensation Added", f"{cond_max:.2f} L/h")

# ------------------------------------------------------------------------------
# TAB 2: SIMULATOR MODE
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("⚙️ Current Operating Conditions")
    colA, colB, colC, colD = st.columns(4)
    curr_grain_rate_sim = colA.number_input("Grain Rate (MT/h)", value=5.0, step=0.5, key="sim_grain")
    curr_rpm_sim = colB.number_input("Conveyor RPM", value=40.0, step=1.0, key="sim_rpm")
    curr_inlet_temp_sim = colC.number_input("Inlet Grain Temp (°C)", value=38.0, step=1.0, key="sim_temp")
    curr_inlet_moist_sim = colD.number_input("Inlet Moisture (%)", value=10.04, step=0.1, key="sim_inlet_moist")
    
    st.markdown("---")

    st.subheader("🕹️ Machine Inputs")
    st.caption("Input your desired steam and water rates to predict what will happen to the grain.")
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_steam = st.number_input("Steam Added (kg/h)", value=250.0, step=10.0, key="sim_steam")
    with sim_col2:
        sim_water = st.number_input("Tempering Water Added (L/h)", value=150.0, step=10.0, key="sim_water")

    if st.button("🔬 Run Forward Simulation", use_container_width=True):
        with st.spinner("Calculating physical state and biological outcomes..."):
            
            results = predict_outcomes_from_inputs(
                steam_rate_kg_h=sim_steam,
                water_added_L_h=sim_water,
                grain_rate_mt_h=curr_grain_rate_sim,
                rpm=curr_rpm_sim,
                inlet_temp=curr_inlet_temp_sim,
                inlet_moist_pct=curr_inlet_moist_sim,
                efficiency=sys_eff,
                moisture_loss_milling=sys_loss,
                enz_params=enz_params,
                glut_params=glut_params
            )
            
            # --- Render Graphs ---
            sim_fig = generate_simulation_fig(
                final_temp=results['final_temperature_c'], 
                moisture=results['tempered_wheat_moisture_pct'], 
                lipase_val=results['lipase_inactivated_pct'], 
                perox_val=results['peroxidase_inactivated_pct'], 
                gluten_val=results['gluten_retained_pct'], 
                enz_params=enz_params, 
                glut_params=glut_params
            )
            st.plotly_chart(sim_fig, use_container_width=True)

            # --- Render Metrics ---
            st.markdown("---")
            st.subheader("📊 Predicted Physical State")
            phys_1, phys_2, phys_3, phys_4 = st.columns(4)
            phys_1.metric("Grain Temperature", f"{results['final_temperature_c']:.1f} °C")
            phys_2.metric("Tempered Moisture", f"{results['tempered_wheat_moisture_pct']:.2f} %")
            phys_3.metric("Final Atta Moisture", f"{results['final_atta_moisture_pct']:.2f} %")
            phys_4.metric("Condensation Added", f"{results['steam_condensed_L_h']:.2f} L/h")
            
            st.markdown("---")
            st.subheader("🧬 Predicted Biological Outcomes")
            bio_1, bio_2, bio_3 = st.columns(3)
            bio_1.metric("Lipase Inactivated", f"{results['lipase_inactivated_pct']:.1f} %")
            bio_2.metric("Peroxidase Inactivated", f"{results['peroxidase_inactivated_pct']:.1f} %")
            bio_3.metric("Dry Gluten Retained", f"{results['gluten_retained_pct']:.2f} %")

# ------------------------------------------------------------------------------
# TAB 3: VISUALIZATION MODE
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🎥 Live Particle & Conditioning Physics")
    st.caption("Adjust the sliders inside this window to view a real-time thermodynamic visualization.")
    components.html(SCM_HTML_ENGINE, height=850, scrolling=False)
