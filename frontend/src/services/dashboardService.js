import axios from 'axios'

const API_URL = (import.meta.env.VITE_API_URL || '');

export const getDashboard = async () => {
  try {
    const token = localStorage.getItem('accessToken')
    const res = await axios.get(`${API_URL}/api/dashboard`, {
      headers: { Authorization: `Bearer ${token}` },
      timeout: 3000,
    })
    return res.data || { totalKendaraan: 0, avgDJ: 0, losCount: {} }
  } catch (error) {
    // Backend tidak aktif, return data dummy
    return { totalKendaraan: 0, avgDJ: 0, losCount: {} }
  }
}
