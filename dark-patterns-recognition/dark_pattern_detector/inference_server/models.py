
from django.db import models

class DetectionResult(models.Model):
    site_url = models.URLField(max_length=400)
    phrase = models.CharField(max_length=1000)
    dark_pattern_type = models.CharField(max_length=100)
    confidence_score = models.FloatField()
    feedback = models.BooleanField(default=False)
    def __str__(self):
        return f"Detection Result for {self.site_url}: {self.phrase}, {self.dark_pattern_type}, Confidence: {self.confidence_score}"
