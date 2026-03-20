from django.db import models

class Prediction(models.Model):
    image_name = models.CharField(max_length=255)
    prediction = models.CharField(max_length=50) # Damaged, Intact, Non-Parcel
    confidence = models.FloatField()
    severity = models.CharField(max_length=50) # Safe, Moderate, Severe, Unknown
    created_at = models.DateTimeField(auto_now_add=True)

    def __cl_repr__(self):
        return f"{self.prediction} ({self.severity})"

    class Meta:
        db_table = 'Predictions'
        ordering = ['-created_at']
