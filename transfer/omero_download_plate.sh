#!/bin/bash
#
# OMERO Data Transfer Script
# 
# This script automates the download of OMERO objects (Plate, Screen, etc.)
# with built-in error handling and automatic reconnection for unstable connections.
# 
# Usage: ./omero_download.sh [OMERO_OBJECT_ID]
# Example: ./omero_download.sh 256  (to download Plate:256)
#
# Before running:
# 1. Edit the configuration section below with your OMERO details
# 2. Set OMERO_PASSWORD environment variable: export OMERO_PASSWORD='your_password'
# 3. Make script executable: chmod +x omero_download.sh
#

# Configuration - Edit these values for your environment
OMERO_USER="your_username"
OMERO_SERVER="omero-t.services.universiteitleiden.nl"
OMERO_PORT="4064"
OMERO_GROUP="LACDR_DDS_water_RA"

# Customize this section based on what you're downloading
OBJECT_TYPE="Plate"  # Options: Plate, Screen, Dataset, Project, etc.
OBJECT_ID=${1:-256}  # Default is 256, but can be overridden from command line
OUTPUT_FOLDER="${OBJECT_TYPE}${OBJECT_ID}.tar"

# Retry settings
MAX_RETRIES=10
RETRY_DELAY=5  # seconds

# Check for password
if [ -z "$OMERO_PASSWORD" ]; then
    echo "OMERO_PASSWORD environment variable not set."
    echo "Set it using: export OMERO_PASSWORD='your_password'"
    echo "Or enter your password now (less secure):"
    read -s OMERO_PASSWORD
    export OMERO_PASSWORD
fi

# Function to log in to OMERO
login_omero() {
    echo "Logging in to OMERO..."
    omero login -s $OMERO_SERVER -p $OMERO_PORT -u $OMERO_USER -w $OMERO_PASSWORD -g $OMERO_GROUP
    if [ $? -ne 0 ]; then
        echo "Failed to log in to OMERO. Check credentials."
        return 1
    fi
    echo "Login successful."
    return 0
}

# Function to download an OMERO object with automatic retries
download_object() {
    local object_type=$1
    local object_id=$2
    local output=$3
    local attempt=0
    local success=false
    local log_file="omero_transfer_${object_type}${object_id}.log"
    
    echo "=== Starting download of $object_type:$object_id ==="
    
    while [ $attempt -lt $MAX_RETRIES ] && [ "$success" = false ]; do
        attempt=$((attempt + 1))
        
        if [ $attempt -gt 1 ]; then
            echo "Retry attempt $attempt of $MAX_RETRIES"
        fi
        
        # Check if we're logged in
        omero sessions who 2>&1 | grep -q $OMERO_USER
        if [ $? -ne 0 ]; then
            echo "No active session, logging in again..."
            login_omero || return 1
        fi
        
        # Run the download command and stream output
        echo "Executing: omero transfer pack $object_type:$object_id $output"
        omero transfer pack $object_type:$object_id $output 2>&1 | tee "$log_file"
        
        # Check the exit status
        transfer_status=${PIPESTATUS[0]}
        
        if [ $transfer_status -eq 0 ]; then
            success=true
            echo "Successfully downloaded $object_type:$object_id"
        else
            # Check log file for specific errors
            if grep -q "ConnectionLostException" "$log_file" || grep -q "Previous session expired" "$log_file"; then
                echo "Connection lost during download, will retry in $RETRY_DELAY seconds..."
                sleep $RETRY_DELAY
                
                # Try to log in again
                login_omero || return 1
            else
                echo "Error occurred during download, will retry in $RETRY_DELAY seconds..."
                sleep $RETRY_DELAY
            fi
        fi
    done
    
    if [ "$success" = false ]; then
        echo "Failed to download $object_type:$object_id after $MAX_RETRIES attempts"
        return 1
    fi
    
    return 0
}

# Main execution
echo "============================================================"
echo "OMERO Resilient Download Script"
echo "Object: $OBJECT_TYPE:$OBJECT_ID"
echo "Output: $OUTPUT_FOLDER"
echo "============================================================"

# Initial login
login_omero || exit 1

# Download the object
download_object $OBJECT_TYPE $OBJECT_ID $OUTPUT_FOLDER

if [ $? -ne 0 ]; then
    echo "ERROR: Could not download $OBJECT_TYPE:$OBJECT_ID"
    exit 1
else
    echo "SUCCESS: $OBJECT_TYPE:$OBJECT_ID successfully downloaded to $OUTPUT_FOLDER"
fi

echo "Download process completed."
