
import re
from urllib.parse import urlparse

class RiskEngine:
    def __init__(self):
        self.suspicious_keywords = [
            'login', 'verify', 'update', 'account', 'banking', 'secure', 'confirm', 
            'signin', 'wallet', 'password', 'admin', 'service', 'support'
        ]

    def calculate_risk_score(self, url, ml_probability):
        """
        Calculate final risk score based on ML probability and rule-based checks.
        Final Score = 70% ML + 30% Rule-based
        """
        rule_score, rule_factors = self._calculate_rule_score(url)
        
        # Convert ML probability (0-1) to 0-100 scale
        ml_score = ml_probability * 100
        
        final_score = (0.7 * ml_score) + (0.3 * rule_score)
        
        factors = rule_factors
        if ml_probability > 0.7:
             factors.append(f"High technical probability ({ml_probability:.2f})")
        
        return final_score, factors

    def _calculate_rule_score(self, url):
        score = 0
        factors = []
        
        # 1. Suspicious Keywords (Max 40 points)
        keyword_hits = [kw for kw in self.suspicious_keywords if kw in url.lower()]
        if keyword_hits:
            score += min(len(keyword_hits) * 15, 40)
            factors.append(f"Suspicious keywords found: {', '.join(keyword_hits)}")
            
        # 2. IP Address Presence (Max 30 points)
        if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', url):
            score += 30
            factors.append("IP address usage in URL")
            
        # 3. Use of HTTP instead of HTTPS (Max 20 points)
        if hasattr(url, 'lower') and not url.lower().startswith('https://'):
             score += 20
             factors.append("Insecure HTTP connection")
             
        # 4. Long URL (Max 10 points)
        if len(url) > 75:
            score += 10
            factors.append("Abnormally long URL")
            
        return min(score, 100), factors

    def determine_risk_level(self, score):
        if score < 40:
            return "Low"
        elif score < 70:
            return "Medium"
        else:
            return "High"
