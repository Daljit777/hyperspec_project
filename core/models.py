from django.db import models
from django.contrib.auth.models import User
import os


class DataFile(models.Model):
    FILE_TYPES = [
        ('he5', 'PRISMA HE5'),
        ('tiff', 'GeoTIFF'),
        ('envi', 'ENVI HDR/IMG'),
    ]

    STATUS_CHOICES = [
        ('ready', 'Ready'),
        ('processing', 'Processing'),
        ('error', 'Error'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='data_files')
    name = models.CharField(max_length=512)
    file = models.FileField(upload_to='uploads/', blank=True, null=True)
    file_path = models.CharField(max_length=1024, blank=True, null=True,
                                 help_text="Path to pre-existing file on disk")
    file_type = models.CharField(max_length=10, choices=FILE_TYPES, default='he5')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='ready')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    # Hyperspectral cube metadata
    rows = models.IntegerField(null=True, blank=True)
    cols = models.IntegerField(null=True, blank=True)
    bands = models.IntegerField(null=True, blank=True)
    vnir_bands = models.IntegerField(null=True, blank=True)
    swir_bands = models.IntegerField(null=True, blank=True)
    wavelengths_vnir = models.JSONField(null=True, blank=True)
    wavelengths_swir = models.JSONField(null=True, blank=True)
    metadata = models.JSONField(null=True, blank=True)
    notes = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return self.name

    def get_absolute_path(self):
        """Return the actual filesystem path to the file."""
        if self.file_path:
            return self.file_path
        if self.file:
            return self.file.path
        return None


class ProcessingJob(models.Model):
    JOB_TYPES = [
        ('radiometric', 'Radiometric Correction'),
        ('geometric', 'Geometric Correction'),
        ('atmospheric', 'Atmospheric Correction'),
        ('index', 'Spectral Index'),
        ('pca', 'PCA Analysis'),
        ('ppi', 'PPI Endmember Extraction'),
        ('sam', 'SAM Target Detection'),
        ('lsu', 'Linear Spectral Unmixing'),
        ('classification', 'Classification'),
        ('clustering', 'Clustering'),
    ]

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('done', 'Done'),
        ('error', 'Error'),
    ]

    data_file = models.ForeignKey(DataFile, on_delete=models.CASCADE, related_name='jobs')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    job_type = models.CharField(max_length=30, choices=JOB_TYPES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    output_file = models.FileField(upload_to='processed/', null=True, blank=True)
    log = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    parameters = models.JSONField(null=True, blank=True)
    result_data = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.job_type} on {self.data_file.name} [{self.status}]"


class SpectralProfile(models.Model):
    """Saved spectral profiles extracted from pixels."""
    data_file = models.ForeignKey(DataFile, on_delete=models.CASCADE, related_name='profiles')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    label = models.CharField(max_length=100, default='Profile')
    row = models.IntegerField()
    col = models.IntegerField()
    spectrum = models.JSONField()        # list of reflectance/radiance values
    wavelengths = models.JSONField()     # list of wavelengths
    sensor_type = models.CharField(max_length=20, default='VNIR')  # VNIR or SWIR
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.label} at ({self.row},{self.col}) - {self.data_file.name}"
