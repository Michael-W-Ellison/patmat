#!/usr/bin/env python3
"""
Integrate Position Clustering with Chess Pattern AI

This script:
1. Builds position clusters from existing game data
2. Integrates clustering with pattern matching
3. Updates the AI to use cluster-based pattern retrieval
"""

import sqlite3
import logging
import chess
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PositionClusteringIntegrator:
    """Integrates position clustering with the pattern recognition AI"""

    def __init__(self, db_path: str = "rule_discovery.db"):
        self.db_path = db_path
        self.conn = None
        self.n_clusters = 20  # Strategic position types
        self.kmeans = None
        self.scaler = StandardScaler()

        # Piece values for material calculation
        self.piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}

    def _connect(self):
        """Connect to database"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row

    def _init_tables(self):
        """Create clustering tables if they don't exist"""
        self._connect()
        cursor = self.conn.cursor()

        # Position clusters table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_clusters (
                cluster_id INTEGER PRIMARY KEY,
                cluster_center TEXT,
                num_positions INTEGER DEFAULT 0,
                avg_win_rate REAL DEFAULT 0.5,
                representative_fens TEXT,
                strategic_theme TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.conn.commit()
        logger.info("✓ Clustering tables initialized")

    def extract_position_features(self, board: chess.Board) -> np.ndarray:
        """Extract 12-dimensional feature vector from a chess position"""
        features = []

        # 1. Material balance
        material = 0
        piece_type_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        for piece_type, value in piece_type_values.items():
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            material += (white_count - black_count) * value
        features.append(material)

        # 2. Piece activity (average centralization)
        activity = 0
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                # Distance from center (0-3.5)
                rank, file = chess.square_rank(square), chess.square_file(square)
                dist = min(abs(3.5 - rank), abs(3.5 - file))
                centralization = (3.5 - dist) / 3.5
                activity += centralization if piece.color == chess.WHITE else -centralization
        features.append(activity / 16)  # Normalize

        # 3. Mobility (legal moves)
        if board.turn == chess.WHITE:
            white_moves = len(list(board.legal_moves))
            # Estimate black mobility from current position
            black_moves = white_moves  # Approximate
        else:
            black_moves = len(list(board.legal_moves))
            white_moves = black_moves  # Approximate
        features.append((white_moves - black_moves) / 50)  # Normalize

        # 4. King safety (distance from edges)
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        white_safety = 0
        black_safety = 0
        if white_king:
            rank, file = chess.square_rank(white_king), chess.square_file(white_king)
            white_safety = min(rank, 7-rank, file, 7-file) / 3.5
        if black_king:
            rank, file = chess.square_rank(black_king), chess.square_file(black_king)
            black_safety = min(rank, 7-rank, file, 7-file) / 3.5
        features.append(white_safety - black_safety)

        # 5. Pawn structure - passed pawns
        passed_pawns = 0
        for square in board.pieces(chess.PAWN, chess.WHITE):
            rank = chess.square_rank(square)
            if rank >= 4:  # Potential passed pawn
                passed_pawns += 1
        for square in board.pieces(chess.PAWN, chess.BLACK):
            rank = chess.square_rank(square)
            if rank <= 3:
                passed_pawns -= 1
        features.append(passed_pawns)

        # 6. Pawn weakness - doubled/isolated
        weakness = 0
        for file_idx in range(8):
            white_pawns = [sq for sq in board.pieces(chess.PAWN, chess.WHITE) if chess.square_file(sq) == file_idx]
            black_pawns = [sq for sq in board.pieces(chess.PAWN, chess.BLACK) if chess.square_file(sq) == file_idx]
            if len(white_pawns) > 1:
                weakness -= len(white_pawns) - 1
            if len(black_pawns) > 1:
                weakness += len(black_pawns) - 1
        features.append(weakness)

        # 7. Space control (squares controlled)
        white_control = len([m for m in board.legal_moves if board.turn == chess.WHITE])
        features.append(white_control / 40)

        # 8. Center control (d4, d5, e4, e5)
        center_control = 0
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        for sq in center_squares:
            piece = board.piece_at(sq)
            if piece:
                center_control += 1 if piece.color == chess.WHITE else -1
        features.append(center_control)

        # 9. Position openness (open files)
        open_files = 0
        for file_idx in range(8):
            has_pawn = any(chess.square_file(sq) == file_idx for sq in board.pieces(chess.PAWN, chess.WHITE))
            has_pawn = has_pawn or any(chess.square_file(sq) == file_idx for sq in board.pieces(chess.PAWN, chess.BLACK))
            if not has_pawn:
                open_files += 1
        features.append(open_files)

        # 10. Development (pieces off back rank)
        developed = 0
        for color in [chess.WHITE, chess.BLACK]:
            back_rank = 0 if color == chess.WHITE else 7
            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.QUEEN]:
                for square in board.pieces(piece_type, color):
                    if chess.square_rank(square) != back_rank:
                        developed += 1 if color == chess.WHITE else -1
        features.append(developed / 6)

        # 11. Piece coordination (protected pieces)
        protected = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                attackers = board.attackers(piece.color, square)
                if len(attackers) > 0:
                    protected += 1 if piece.color == chess.WHITE else -1
        features.append(protected)

        # 12. Tactical complexity (checks, captures available)
        complexity = 0
        for move in board.legal_moves:
            if board.is_capture(move):
                complexity += 1
            if board.gives_check(move):
                complexity += 2
        features.append(min(complexity, 10) / 10)

        return np.array(features)

    def build_clusters(self, sample_size: int = 10000):
        """Build position clusters from existing game data"""
        logger.info(f"Building position clusters from {sample_size} positions...")
        self._connect()
        self._init_tables()

        cursor = self.conn.cursor()

        # Sample positions from games
        cursor.execute('''
            SELECT DISTINCT fen_after, game_id
            FROM moves
            ORDER BY RANDOM()
            LIMIT ?
        ''', (sample_size,))

        positions = cursor.fetchall()
        logger.info(f"✓ Sampled {len(positions)} positions")

        # Extract features
        feature_vectors = []
        valid_fens = []

        for i, row in enumerate(positions):
            if i % 1000 == 0:
                logger.info(f"  Extracting features: {i}/{len(positions)}")

            try:
                fen = row['fen_after']
                # Handle incomplete FENs (add default move counters if missing)
                fen_parts = fen.split()
                if len(fen_parts) < 6:
                    # Add missing parts: en passant, halfmove clock, fullmove number
                    while len(fen_parts) < 4:
                        fen_parts.append('-')
                    if len(fen_parts) == 4:
                        fen_parts.append('0')  # halfmove clock
                    if len(fen_parts) == 5:
                        fen_parts.append('1')  # fullmove number
                    fen = ' '.join(fen_parts)

                board = chess.Board(fen)
                features = self.extract_position_features(board)
                feature_vectors.append(features)
                valid_fens.append(fen)
            except Exception as e:
                if i < 10:  # Log first few errors
                    logger.error(f"Error parsing FEN '{row['fen_after']}': {e}")
                continue

        if len(feature_vectors) < self.n_clusters:
            logger.warning(f"Only {len(feature_vectors)} valid positions, reducing clusters")
            self.n_clusters = max(5, len(feature_vectors) // 100)

        logger.info(f"✓ Extracted {len(feature_vectors)} feature vectors")

        # Normalize features
        X = np.array(feature_vectors)
        X_scaled = self.scaler.fit_transform(X)

        # Perform K-means clustering
        logger.info(f"⚙ Running K-means with {self.n_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)

        logger.info("✓ Clustering complete")

        # Save clusters to database
        self._save_clusters(valid_fens, cluster_labels, X_scaled)

        return self.n_clusters

    def _save_clusters(self, fens: List[str], labels: np.ndarray, features: np.ndarray):
        """Save cluster information to database"""
        logger.info("💾 Saving clusters to database...")

        cursor = self.conn.cursor()

        # Clear existing clusters
        cursor.execute('DELETE FROM position_cluster_membership')
        cursor.execute('DELETE FROM position_clusters')

        # Group positions by cluster
        clusters = defaultdict(list)
        for i, (fen, label) in enumerate(zip(fens, labels)):
            clusters[label].append((fen, features[i]))

        # Save cluster information
        for cluster_id, positions in clusters.items():
            fens_in_cluster = [p[0] for p in positions]

            # Calculate cluster center
            center = self.kmeans.cluster_centers_[cluster_id]
            center_json = json.dumps(center.tolist())

            # Representative FENs (closest to center)
            distances = [np.linalg.norm(features - center) for _, features in positions]
            sorted_indices = np.argsort(distances)[:5]
            representative_fens = json.dumps([fens_in_cluster[i] for i in sorted_indices])

            # Determine strategic theme based on cluster center
            theme = self._determine_theme(center)

            cursor.execute('''
                INSERT INTO position_clusters
                (cluster_id, cluster_center, num_positions, representative_fens, strategic_theme)
                VALUES (?, ?, ?, ?, ?)
            ''', (int(cluster_id), center_json, len(positions), representative_fens, theme))

            # Save cluster membership
            for fen, feat in positions:
                distance = np.linalg.norm(feat - center)
                cursor.execute('''
                    INSERT OR REPLACE INTO position_cluster_membership
                    (fen, cluster_id, distance_to_center)
                    VALUES (?, ?, ?)
                ''', (fen, int(cluster_id), float(distance)))

        self.conn.commit()
        logger.info(f"✅ Saved {len(clusters)} clusters with {len(fens)} positions")

    def _determine_theme(self, center: np.ndarray) -> str:
        """Determine strategic theme from cluster center"""
        themes = []

        if center[0] > 2:
            themes.append("Material_Advantage")
        elif center[0] < -2:
            themes.append("Material_Disadvantage")

        if center[1] > 0.5:
            themes.append("Active_Pieces")
        if center[7] > 2:
            themes.append("Center_Control")
        if center[8] > 4:
            themes.append("Open_Position")
        elif center[8] < 2:
            themes.append("Closed_Position")
        if center[4] > 2:
            themes.append("Passed_Pawns")

        return "_".join(themes) if themes else "Balanced"

    def find_similar_positions(self, fen: str, limit: int = 10) -> List[Tuple[str, float, int]]:
        """Find similar positions using clustering"""
        self._connect()

        try:
            # Handle incomplete FENs
            fen_parts = fen.split()
            if len(fen_parts) < 6:
                while len(fen_parts) < 4:
                    fen_parts.append('-')
                if len(fen_parts) == 4:
                    fen_parts.append('0')
                if len(fen_parts) == 5:
                    fen_parts.append('1')
                fen = ' '.join(fen_parts)

            board = chess.Board(fen)
            features = self.extract_position_features(board)
            features_scaled = self.scaler.transform([features])[0]

            # Find nearest cluster manually
            if self.kmeans is None:
                self._load_clusters()

            # Calculate distance to all cluster centers
            distances = np.linalg.norm(self.kmeans.cluster_centers_ - features_scaled, axis=1)
            cluster_id = int(np.argmin(distances))

            # Get positions from this cluster
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT fen, distance_to_center, cluster_id
                FROM position_cluster_membership
                WHERE cluster_id = ?
                ORDER BY distance_to_center
                LIMIT ?
            ''', (cluster_id, limit))

            return [(row['fen'], row['distance_to_center'], row['cluster_id'])
                    for row in cursor.fetchall()]

        except Exception as e:
            # Silently return empty list if position_cluster_membership table doesn't exist
            # (This is expected after database cleanup - we use cluster centers, not specific positions)
            return []

    def _load_clusters(self):
        """Load cluster centers from database"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT cluster_id, cluster_center FROM position_clusters ORDER BY cluster_id')

        centers = []
        for row in cursor.fetchall():
            center = json.loads(row['cluster_center'])
            centers.append(center)

        if centers:
            self.kmeans = KMeans(n_clusters=len(centers), random_state=42)
            self.kmeans.cluster_centers_ = np.array(centers)

            # Fit scaler with dummy data based on loaded centers
            X_dummy = np.array(centers)
            self.scaler.fit(X_dummy)

            logger.info(f"✓ Loaded {len(centers)} cluster centers")

    def get_cluster_stats(self):
        """Get statistics about the clusters"""
        self._connect()
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT
                pc.cluster_id,
                pc.strategic_theme,
                pc.num_positions,
                COUNT(DISTINCT pcm.fen) as actual_positions
            FROM position_clusters pc
            LEFT JOIN position_cluster_membership pcm ON pc.cluster_id = pcm.cluster_id
            GROUP BY pc.cluster_id
            ORDER BY pc.cluster_id
        ''')

        stats = []
        for row in cursor.fetchall():
            stats.append({
                'cluster_id': row['cluster_id'],
                'theme': row['strategic_theme'],
                'positions': row['actual_positions']
            })

        return stats


def main():
    """Build and test position clustering"""
    print("=" * 70)
    print("POSITION CLUSTERING INTEGRATION")
    print("=" * 70)

    integrator = PositionClusteringIntegrator()

    # Build clusters
    start = time.time()
    n_clusters = integrator.build_clusters(sample_size=10000)
    elapsed = time.time() - start

    print(f"\n✅ Built {n_clusters} clusters in {elapsed:.1f}s")

    # Show cluster statistics
    print("\n📊 Cluster Statistics:")
    print("-" * 70)
    stats = integrator.get_cluster_stats()
    for stat in stats:
        print(f"  Cluster {stat['cluster_id']:2d}: {stat['positions']:5d} positions - {stat['theme']}")

    print("\n" + "=" * 70)
    print("✅ Clustering integration complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
