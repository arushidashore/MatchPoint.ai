# Cache Manager for MatchPoint.ai
import os
import json
import hashlib
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

class CacheManager:
    def __init__(self, cache_dir: str = 'cache', db_path: str = 'cache.db'):
        self.cache_dir = cache_dir
        self.db_path = db_path
        self.max_cache_size = 1000  # Maximum number of cached entries
        self.max_cache_age = 7 * 24 * 3600  # 7 days in seconds
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Clean old cache entries on startup
        self.cleanup_old_cache()
    
    def _init_database(self):
        """Initialize the cache database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_hash TEXT UNIQUE NOT NULL,
                height REAL NOT NULL,
                stroke_type TEXT NOT NULL,
                feedback TEXT NOT NULL,
                video_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_video_hash ON cache_entries(video_hash)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)
        ''')
        
        conn.commit()
        conn.close()
    
    def _generate_video_hash(self, video_path: str, height: float, stroke_type: str) -> str:
        """Generate a hash for the video and parameters."""
        try:
            # Get file stats
            stat = os.stat(video_path)
            file_size = stat.st_size
            mtime = stat.st_mtime
            
            # Create hash from file properties and parameters
            hash_input = f"{file_size}_{mtime}_{height}_{stroke_type}"
            return hashlib.sha256(hash_input.encode()).hexdigest()
        except Exception as e:
            logging.error(f"Error generating video hash: {e}")
            return hashlib.sha256(f"{video_path}_{height}_{stroke_type}_{time.time()}".encode()).hexdigest()
    
    def get_cached_result(self, video_path: str, height: float, stroke_type: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available."""
        video_hash = self._generate_video_hash(video_path, height, stroke_type)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT feedback, video_path, created_at FROM cache_entries 
            WHERE video_hash = ? AND height = ? AND stroke_type = ?
        ''', (video_hash, height, stroke_type))
        
        result = cursor.fetchone()
        
        if result:
            feedback, cached_video_path, created_at = result
            
            # Check if video file still exists
            if os.path.exists(cached_video_path):
                # Update access time and count
                cursor.execute('''
                    UPDATE cache_entries 
                    SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                    WHERE video_hash = ?
                ''', (video_hash,))
                conn.commit()
                
                logging.info(f"Cache hit for video hash: {video_hash}")
                conn.close()
                
                return {
                    'feedback': feedback,
                    'video_path': cached_video_path,
                    'cached': True,
                    'cache_age': time.time() - datetime.fromisoformat(created_at).timestamp()
                }
            else:
                # Remove stale cache entry
                cursor.execute('DELETE FROM cache_entries WHERE video_hash = ?', (video_hash,))
                conn.commit()
        
        conn.close()
        return None
    
    def cache_result(self, video_path: str, height: float, stroke_type: str, 
                    feedback: str, output_video_path: str) -> bool:
        """Cache the analysis result."""
        try:
            video_hash = self._generate_video_hash(video_path, height, stroke_type)
            file_size = os.path.getsize(output_video_path)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Remove old cache entry if exists
            cursor.execute('DELETE FROM cache_entries WHERE video_hash = ?', (video_hash,))
            
            # Insert new cache entry
            cursor.execute('''
                INSERT INTO cache_entries 
                (video_hash, height, stroke_type, feedback, video_path, file_size, created_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
            ''', (video_hash, height, stroke_type, feedback, output_video_path, file_size))
            
            conn.commit()
            conn.close()
            
            logging.info(f"Cached result for video hash: {video_hash}")
            return True
            
        except Exception as e:
            logging.error(f"Error caching result: {e}")
            return False
    
    def cleanup_old_cache(self):
        """Clean up old cache entries and files."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get old cache entries
            cutoff_time = datetime.now() - timedelta(seconds=self.max_cache_age)
            cursor.execute('''
                SELECT video_path FROM cache_entries 
                WHERE created_at < ? OR last_accessed < ?
            ''', (cutoff_time.isoformat(), cutoff_time.isoformat()))
            
            old_entries = cursor.fetchall()
            
            # Remove old files and database entries
            for (video_path,) in old_entries:
                try:
                    if os.path.exists(video_path):
                        os.remove(video_path)
                    cursor.execute('DELETE FROM cache_entries WHERE video_path = ?', (video_path,))
                except Exception as e:
                    logging.error(f"Error removing old cache file {video_path}: {e}")
            
            # Limit cache size
            cursor.execute('SELECT COUNT(*) FROM cache_entries')
            count = cursor.fetchone()[0]
            
            if count > self.max_cache_size:
                # Remove least recently used entries
                cursor.execute('''
                    DELETE FROM cache_entries 
                    WHERE id IN (
                        SELECT id FROM cache_entries 
                        ORDER BY last_accessed ASC 
                        LIMIT ?
                    )
                ''', (count - self.max_cache_size,))
            
            conn.commit()
            conn.close()
            
            logging.info(f"Cleaned up {len(old_entries)} old cache entries")
            
        except Exception as e:
            logging.error(f"Error cleaning up cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM cache_entries')
            total_entries = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(file_size) FROM cache_entries')
            total_size = cursor.fetchone()[0] or 0
            
            cursor.execute('''
                SELECT AVG(access_count) FROM cache_entries 
                WHERE created_at > datetime('now', '-1 day')
            ''')
            avg_access = cursor.fetchone()[0] or 0
            
            cursor.execute('''
                SELECT COUNT(*) FROM cache_entries 
                WHERE created_at > datetime('now', '-1 day')
            ''')
            recent_entries = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_entries': total_entries,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'avg_access_count': round(avg_access, 2),
                'recent_entries_24h': recent_entries,
                'cache_hit_rate': self._calculate_hit_rate()
            }
            
        except Exception as e:
            logging.error(f"Error getting cache stats: {e}")
            return {}
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT AVG(access_count) FROM cache_entries')
            avg_access = cursor.fetchone()[0] or 0
            
            conn.close()
            
            # Simple hit rate calculation based on access count
            return min(round(avg_access * 10, 2), 100.0)
            
        except Exception:
            return 0.0
    
    def clear_cache(self):
        """Clear all cache entries and files."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all cached video paths
            cursor.execute('SELECT video_path FROM cache_entries')
            video_paths = cursor.fetchall()
            
            # Remove all video files
            for (video_path,) in video_paths:
                try:
                    if os.path.exists(video_path):
                        os.remove(video_path)
                except Exception as e:
                    logging.error(f"Error removing cache file {video_path}: {e}")
            
            # Clear database
            cursor.execute('DELETE FROM cache_entries')
            conn.commit()
            conn.close()
            
            logging.info("Cache cleared successfully")
            
        except Exception as e:
            logging.error(f"Error clearing cache: {e}")

# Global cache manager instance
cache_manager = CacheManager()

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return cache_manager
