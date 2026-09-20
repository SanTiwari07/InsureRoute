import { motion, animate } from 'framer-motion'
import { useEffect, useRef } from 'react'
<<<<<<< HEAD
import { AlertTriangle, Map, Navigation, Cloud } from 'lucide-react'
=======
import { AlertTriangle, Shield, Cloud, PiggyBank } from 'lucide-react'
>>>>>>> bab34a3337a6c33e3f1fd3813bf6d277780772d4

function AnimatedNumber({ value, prefix = '', suffix = '', decimals = 1, className = '' }) {
  const ref = useRef(null)
  const prev = useRef(0)

  useEffect(() => {
    const node = ref.current
    if (!node) return
    const from = prev.current
    prev.current = value
    const ctrl = animate(from, value, {
      duration: 0.9,
      ease: [0.4, 0, 0.2, 1],
      onUpdate(v) {
        node.textContent = prefix + v.toFixed(decimals) + suffix
      },
    })
    return () => ctrl.stop()
  }, [value, prefix, suffix, decimals])

  return <span ref={ref} className={className}>{prefix}{value.toFixed(decimals)}{suffix}</span>
}

const CARDS = [
  {
<<<<<<< HEAD
    key: 'weather',
    label: 'Weather',
    suffix: '%',
    icon: Cloud,
    accent: 'border-t-blue-500',
    iconColor: 'text-blue-500',
    desc: 'severity',
    threshold: (v) => v > 50 ? 'danger' : v > 20 ? 'warning' : 'success',
  },
  {
    key: 'traffic',
    label: 'Traffic Delay',
    suffix: 'x',
    prefix: '',
    icon: Navigation,
    accent: 'border-t-warning',
    iconColor: 'text-warning',
    desc: 'ratio',
    threshold: (v) => v > 2 ? 'danger' : v > 1.2 ? 'warning' : 'success',
  },
  {
    key: 'distance',
    label: 'Distance',
    suffix: ' km',
    icon: Map,
    accent: 'border-t-success',
    iconColor: 'text-success',
    desc: 'total route length',
    threshold: () => 'success',
  },
  {
    key: 'risk',
    label: 'Current Risk',
    suffix: '%',
    icon: AlertTriangle,
    accent: 'border-t-danger',
    iconColor: 'text-danger',
    desc: 'disruption probability',
    threshold: (v) => v > 60 ? 'danger' : v > 30 ? 'warning' : 'success',
  },
=======
    key: 'cargo_value', // Map to a new key or use 'sla' if parent passes it
    label: 'Gross Cargo Value Shielded',
    suffix: '',
    prefix: '₹',
    icon: Shield,
    accent: 'border-t-primary',
    iconColor: 'text-primary',
    desc: 'At-risk capital secured',
    threshold: () => 'success',
  },
  {
    key: 'co2',
    label: 'Verified Scope 3 CO2 Offset',
    suffix: ' kg',
    prefix: '',
    icon: Cloud,
    accent: 'border-t-success',
    iconColor: 'text-success',
    desc: 'Equivalent to 3 trucks off road',
    threshold: () => 'success',
  },
  {
    key: 'premium_savings',
    label: 'Actuarial Premium Arbitrage',
    suffix: '',
    prefix: '₹',
    icon: PiggyBank,
    accent: 'border-t-success',
    iconColor: 'text-success',
    desc: 'Real-time reduction applied',
    threshold: () => 'success',
  },
  {
    key: 'shocks_averted',
    label: 'Catastrophic Shocks Averted',
    suffix: '',
    prefix: '',
    icon: AlertTriangle,
    accent: 'border-t-warning',
    iconColor: 'text-warning',
    desc: 'High-severity disruptions bypassed',
    threshold: () => 'warning',
  },
>>>>>>> bab34a3337a6c33e3f1fd3813bf6d277780772d4
]

const LEVEL_STYLES = {
  danger:  'text-danger',
  warning: 'text-warning',
  success: 'text-text', // Keep normal text for success looking metrics instead of neon green
}

export default function KPICards({ kpis }) {
  const safeKpis = kpis || { cargo_value: 0, co2: 0, premium_savings: 0, shocks_averted: 0 }

  return (
    <div className="flex gap-3 md:gap-4 w-full">
      {CARDS.map((card, i) => {
        const value = safeKpis[card.key] ?? 0
        const level = card.threshold(value)
        const Icon  = card.icon
        
        return (
          <motion.div
            key={card.key}
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05, duration: 0.4 }}
            className={`glass p-4 md:p-5 flex flex-col justify-center border-t-2 ${card.accent} flex-1 min-h-[100px]`}
          >
            <div className="flex items-center gap-2 mb-1">
              <Icon size={16} className={card.iconColor} />
              <span className="text-sm font-semibold text-slate-600">
                {card.label}
              </span>
            </div>
            
            <div className={`text-3xl md:text-4xl font-black tracking-tight ${LEVEL_STYLES[level]}`}>
              <AnimatedNumber
                value={value}
                prefix={card.prefix || ''}
                suffix={card.suffix}
                decimals={value > 100 ? 0 : 1}
              />
            </div>
            
            <span className="text-xs font-medium text-muted mt-1">
              {card.key === 'co2' ? `Equivalent to ${Math.max(1, Math.round(value / 4000))} trucks off road` : card.desc}
            </span>
          </motion.div>
        )
      })}
    </div>
  )
}
